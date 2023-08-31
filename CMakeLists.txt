cmake_minimum_required(VERSION 3.0.2)
project(ground_sam)

find_package(
  catkin REQUIRED
  COMPONENTS sensor_msgs
             cv_bridge
             rospy
)

catkin_package(
  #  INCLUDE_DIRS include
  #  LIBRARIES yolact_ros
  CATKIN_DEPENDS
    sensor_msgs
    cv_bridge
    rospy
  #  DEPENDS system_lib
)

install(
  PROGRAMS scripts/segmentation.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
