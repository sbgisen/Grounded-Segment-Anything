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

import typing

import cv2
import cv_bridge
import numpy as np
import rospkg
import rospy
import supervision as sv
import torch
import torchvision
from ground_sam.srv import Segmentation as SegmentationSrv
from ground_sam.srv import SegmentationRequest
from ground_sam.srv import SegmentationResponse
from groundingdino.util.inference import Model
from LightHQSAM.setup_light_hqsam import setup_model
from pcl_msgs.msg import PointIndices
from sensor_msgs.msg import Image
from std_msgs.msg import String
from supervision.detection.core import Detections

from segment_anything import SamPredictor


class Segmentation(object):

    def __init__(self) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # GroundingDINO config and checkpoint
        pkg_path = rospkg.RosPack().get_path('ground_sam')
        grounding_dino_config = pkg_path + '/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        grounding_dino_model_path = pkg_path + '/config/groundingdino_swint_ogc.pth'

        # Building GroundingDINO inference model
        self.__grounding_dino_model = Model(model_config_path=grounding_dino_config,
                                            model_checkpoint_path=grounding_dino_model_path)

        # Building MobileSAM predictor
        hqsam_model_path = pkg_path + '/config/sam_hq_vit_tiny.pth'
        checkpoint = torch.load(hqsam_model_path)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=device)

        self.__sam_predictor = SamPredictor(light_hqsam)

        # Predict classes and hyper-param for GroundingDINO
        self.__classes = ['cup_noodles']
        self.__box_threshold = 0.25
        self.__text_threshold = 0.25
        self.__nms_threshold = 0.8

        self.__bridge = cv_bridge.CvBridge()

        rospy.Subscriber('classes', String, self.__classes_callback)
        rospy.Subscriber('~image', Image, self.__callback)
        rospy.Service('~segmentation', SegmentationSrv, self.__srv_callback)
        self.__vis_pub = rospy.Publisher('~vis', Image, queue_size=1)

    def __srv_callback(self, req: SegmentationRequest) -> SegmentationResponse:
        self.__classes = req.classes
        points = np.array([[p.x, p.y] for p in req.points])
        img = self.__bridge.imgmsg_to_cv2(req.image, desired_encoding='bgr8')
        detections = None
        if len(self.__classes) != 0:
            detections = self.__grounding_dino_model.predict_with_classes(image=img,
                                                                          classes=self.__classes,
                                                                          box_threshold=self.__box_threshold,
                                                                          text_threshold=self.__text_threshold)
            box_annotator = sv.BoxAnnotator()

            labels = [f'{self.__classes[class_id]} {confidence:0.2f}' for _, _, confidence, class_id, _ in detections]
            box_annotator.annotate(scene=img.copy(), detections=detections, labels=labels)
            rospy.loginfo(f'Before NMS: {len(detections.xyxy)} boxes')

            nms_idx = torchvision.ops.nms(torch.from_numpy(detections.xyxy), torch.from_numpy(detections.confidence),
                                          self.__nms_threshold).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

            rospy.loginfo(f'After NMS: {len(detections.xyxy)} boxes')

        # convert detections to masks
        xyxy = detections.xyxy if detections is not None else None
        mask = self.__segment(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), xyxy=xyxy, points=points)
        if detections is not None:
            detections.mask = mask
        else:
            xyxy = []
            for m in mask:
                mask_index = np.where(m)
                xyxy.append([mask_index[1].min(), mask_index[0].min(), mask_index[1].max(), mask_index[0].max()])
            detections = Detections(np.array(xyxy))
            detections.mask = mask
            detections.confidence = np.ones(len(xyxy))
            detections.class_id = np.ones(len(xyxy), dtype=np.int16)
            self.__classes = ['unknown']

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [f'{self.__classes[class_id]} {confidence:0.2f}' for _, _, confidence, class_id, _ in detections]
        annotated_image = mask_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        self.__vis_pub.publish(self.__bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8'))

        resp = SegmentationResponse()
        resp.labels.header = req.image.header
        resp.labels.classifier = 'Grounded SAM'
        resp.labels.label_names = labels
        resp.labels.labels = detections.class_id.tolist()
        resp.labels.label_proba = detections.confidence.tolist()
        resp.labels.target_names = req.classes
        resp.indices.header = req.image.header

        resp.indices.cluster_indices = [
            PointIndices(header=req.image.header, indices=np.where(mask.flatten())[0])
            for _, mask, _, _, _ in detections
        ]

        return resp

    def __classes_callback(self, msg: String) -> None:
        self.__classes = msg.data.split(',')

    def __callback(self, msg: Image) -> None:
        img = self.__bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        detections = self.__grounding_dino_model.predict_with_classes(image=img,
                                                                      classes=self.__classes,
                                                                      box_threshold=self.__box_threshold,
                                                                      text_threshold=self.__text_threshold)
        box_annotator = sv.BoxAnnotator()

        labels = [f'{self.__classes[class_id]} {confidence:0.2f}' for _, _, confidence, class_id, _ in detections]
        box_annotator.annotate(scene=img.copy(), detections=detections, labels=labels)
        rospy.loginfo(f'Before NMS: {len(detections.xyxy)} boxes')

        nms_idx = torchvision.ops.nms(torch.from_numpy(detections.xyxy), torch.from_numpy(detections.confidence),
                                      self.__nms_threshold).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        rospy.loginfo(f'After NMS: {len(detections.xyxy)} boxes')

        # convert detections to masks
        detections.mask = self.__segment(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), xyxy=detections.xyxy)

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [f'{self.__classes[class_id]} {confidence:0.2f}' for _, _, confidence, class_id, _ in detections]
        annotated_image = mask_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        self.__vis_pub.publish(self.__bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8'))

    # Prompting SAM with detected boxes
    def __segment(self,
                  image: np.ndarray,
                  xyxy: typing.Optional[np.ndarray] = None,
                  points: typing.Optional[np.ndarray] = None) -> np.ndarray:
        self.__sam_predictor.set_image(image)
        result_masks = []
        if xyxy is not None:
            for box in xyxy:
                masks, scores, logits = self.__sam_predictor.predict(
                    box=box,
                    multimask_output=False,
                    hq_token_only=True,
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
        else:
            masks, scores, logits = self.__sam_predictor.predict(
                point_coords=points,
                point_labels=np.zeros(len(points)),
                multimask_output=False,
                hq_token_only=True,
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)


if __name__ == '__main__':
    rospy.init_node('segmentation')
    segmentation = Segmentation()
    rospy.spin()
