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

import cv2
import cv_bridge
import message_filters
import numpy as np
import rospkg
import rospy
import supervision as sv
import torch
import torchvision
from efficient_sam.build_efficient_sam import build_efficient_sam
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_msgs.msg import RectArray
from sensor_msgs.msg import Image
from supervision.detection.core import Detections


class Segmentation(object):

    def __init__(self) -> None:
        # GroundingDINO config and checkpoint
        pkg_path = rospkg.RosPack().get_path('ground_sam')

        # Building MobileSAM predictor
        efficient_sam_model_path = pkg_path + '/config/efficient_sam_vitt.pt'
        self.__model = build_efficient_sam(encoder_patch_embed_dim=192,
                                           encoder_num_heads=3,
                                           checkpoint=efficient_sam_model_path).cuda().eval()

        self.__nms_threshold = 0.8

        self.__bridge = cv_bridge.CvBridge()

        image_sub = message_filters.Subscriber('~image', Image)
        rect_sub = message_filters.Subscriber('~rects', RectArray)
        result_sub = message_filters.Subscriber('~class', ClassificationResult)
        self.__sync = message_filters.TimeSynchronizer([image_sub, rect_sub, result_sub], 10)
        self.__sync.registerCallback(self.__callback)
        self.__vis_pub = rospy.Publisher('~vis', Image, queue_size=1)

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
        result_masks = []
        for box in detections.xyxy:
            mask = self.__efficient_sam_box_prompt_segment(img, box)
            result_masks.append(mask)

        detections.mask = np.array(result_masks)

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        results_msg.target_names
        labels = [
            f'{label} {confidence:0.2f}'
            for (_, _, confidence, _, _), label in zip(detections, results_msg.label_names)
        ]
        annotated_image = mask_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        self.__vis_pub.publish(self.__bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8'))

    def __efficient_sam_box_prompt_segment(self, image: np.ndarray, pts_sampled: np.ndarray) -> torch.Tensor:
        bbox = torch.reshape(torch.tensor(pts_sampled), [1, 1, 2, 2])
        bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torchvision.transforms.ToTensor()(image)

        predicted_logits, predicted_iou = self.__model(
            img_tensor[None, ...].cuda(),
            bbox.cuda(),
            bbox_labels.cuda(),
        )
        predicted_logits = predicted_logits.cpu()
        all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
        predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

        max_predicted_iou = -1
        selected_mask_using_predicted_iou = None
        for m in range(all_masks.shape[0]):
            curr_predicted_iou = predicted_iou[m]
            if (curr_predicted_iou > max_predicted_iou or selected_mask_using_predicted_iou is None):
                max_predicted_iou = curr_predicted_iou
                selected_mask_using_predicted_iou = all_masks[m]
        return selected_mask_using_predicted_iou


if __name__ == '__main__':
    rospy.init_node('segmentation')
    segmentation = Segmentation()
    rospy.spin()
