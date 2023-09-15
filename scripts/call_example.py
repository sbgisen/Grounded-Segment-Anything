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

import rospy
from geometry_msgs.msg import Point
from ground_sam.srv import Segmentation as SegmentationSrv
from ground_sam.srv import SegmentationRequest
from jsk_recognition_msgs.msg import ClusterPointIndices
from sensor_msgs.msg import Image
from std_msgs.msg import String


class CallSegmentation(object):

    def __init__(self) -> None:
        self.__indices_pub = rospy.Publisher('~cluster_indices', ClusterPointIndices, queue_size=1)
        self.__segmentation = rospy.ServiceProxy('~segmentation', SegmentationSrv)
        self.__segmentation.wait_for_service()
        rospy.Subscriber('~class', String, self.__class_callback)
        rospy.Subscriber('~point', Point, self.__point_callback)

    def __class_callback(self, msg: String) -> None:
        req = SegmentationRequest()
        req.classes = msg.data.split(',')
        img = rospy.wait_for_message('~image', Image)
        req.image = img
        res = self.__segmentation(req)
        self.__indices_pub.publish(res.indices)

    def __point_callback(self, msg: Point) -> None:
        req = SegmentationRequest()
        req.points = [msg]
        img = rospy.wait_for_message('~image', Image)
        req.image = img
        res = self.__segmentation(req)
        self.__indices_pub.publish(res.indices)


if __name__ == '__main__':
    rospy.init_node('call_seg')
    segmentation = CallSegmentation()
    rospy.spin()
