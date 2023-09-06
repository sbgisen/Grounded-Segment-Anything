cmake_minimum_required(VERSION 3.0.2)
project(ground_sam)

find_package(
  catkin REQUIRED
  COMPONENTS jsk_recognition_msgs
             sensor_msgs
             cv_bridge
             rospy
             message_generation
)

add_service_files(FILES Segmentation.srv)

generate_messages(DEPENDENCIES jsk_recognition_msgs sensor_msgs)

catkin_package(
  #  INCLUDE_DIRS include
  #  LIBRARIES yolact_ros
  CATKIN_DEPENDS
    jsk_recognition_msgs
    sensor_msgs
    cv_bridge
    rospy
    message_runtime
  #  DEPENDS system_lib
)

install(
  PROGRAMS scripts/segmentation.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
