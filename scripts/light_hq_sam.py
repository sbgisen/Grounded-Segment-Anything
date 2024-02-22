#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-

# Copyright (c) 2023 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cv_bridge
import message_filters
import numpy as np
import rospkg
import rospy
import supervision as sv
import torch
import torchvision
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import RectArray
from LightHQSAM.setup_light_hqsam import setup_model
from mask_rcnn_ros.msg import Result
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from supervision.detection.core import Detections

from segment_anything import SamPredictor


class Segmentation(object):

    def __init__(self) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pkg_path = rospkg.RosPack().get_path('ground_sam')

        hqsam_model_path = pkg_path + '/config/sam_hq_vit_tiny.pth'
        checkpoint = torch.load(hqsam_model_path)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=device)

        self.__sam_predictor = SamPredictor(light_hqsam)

        self.__nms_threshold = 0.8

        self.__bridge = cv_bridge.CvBridge()

        image_sub = message_filters.Subscriber('~image', Image)
        rect_sub = message_filters.Subscriber('~rects', RectArray)
        result_sub = message_filters.Subscriber('~class', ClassificationResult)
        self.__sync = message_filters.TimeSynchronizer([image_sub, rect_sub, result_sub], 10)
        self.__sync.registerCallback(self.__callback)
        self.__vis_pub = rospy.Publisher('~vis', Image, queue_size=1)
        self.__results_pub = rospy.Publisher('~results', Result, queue_size=1)

    def __callback(self, img_msg: Image, rects_msg: RectArray, results_msg: ClassificationResult) -> None:
        img = self.__bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        if len(rects_msg.rects) == 0:
            rospy.loginfo('No boxes detected')
            detections = Detections.empty()
        else:
            detections = Detections(
                xyxy=np.array([[rect.x, rect.y, rect.x + rect.width, rect.y + rect.height]
                               for rect in rects_msg.rects],
                              dtype=np.float32),
                confidence=np.array(results_msg.label_proba, dtype=np.float32),
                class_id=np.array(results_msg.labels, dtype=int),
            )
        labels = results_msg.label_names
        rospy.loginfo(f'Before NMS: {len(detections.xyxy)} boxes')

        nms_idx = torchvision.ops.nms(torch.from_numpy(detections.xyxy), torch.from_numpy(detections.confidence),
                                      self.__nms_threshold).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        rospy.loginfo(f'After NMS: {len(detections.xyxy)} boxes')

        # convert detections to masks
        self.__sam_predictor.set_image(img)
        result_masks = []
        for box in detections.xyxy:
            masks, scores, logits = self.__sam_predictor.predict(
                box=box,
                multimask_output=False,
                hq_token_only=True,
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])

        detections.mask = np.array(result_masks)

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        results_msg.target_names
        labels = [
            f'{label} {confidence:0.2f}'
            for (_, _, confidence, _, _, _), label in zip(detections, results_msg.label_names)
        ]
        annotated_image = mask_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        self.__vis_pub.publish(self.__bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8'))

        msg_results = Result(header=img_msg.header)
        bridge = cv_bridge.CvBridge()
        for rect, mask, class_id, conf, name in zip(rects_msg.rects, detections.mask, detections.class_id,
                                                    detections.confidence, results_msg.label_names):
            roi = RegionOfInterest()
            roi.x_offset = int(np.clip(rect.x, 0, img.shape[1]))
            roi.y_offset = int(np.clip(rect.y, 0, img.shape[0]))
            roi.width = int(np.clip(rect.width, 0, img.shape[1]))
            roi.height = int(np.clip(rect.height, 0, img.shape[0]))
            msg_results.boxes.append(roi)
            msg_results.class_ids.append(class_id)
            msg_results.scores.append(conf)
            msg_results.class_names.append(name)
            mask_msg = bridge.cv2_to_imgmsg(mask.astype(np.uint8) * 255, encoding='mono8')
            msg_results.masks.append(mask_msg)
        self.__results_pub.publish(msg_results)


if __name__ == '__main__':
    rospy.init_node('segmentation')
    segmentation = Segmentation()
    rospy.spin()
