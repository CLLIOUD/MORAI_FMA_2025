cmake_minimum_required(VERSION 3.0.2)
project(morai_script)

## ROS 패키지 의존성
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  nav_msgs
  tf
  morai_msgs
  custom_msgs
)

catkin_package()

## 파이썬 노드 설치 (rosrun/roslaunch에서 바로 실행되도록)
catkin_install_python(PROGRAMS
  scripts/path.py
  scripts/path_mux.py
  scripts/status.py
  scripts/traffic_stop.py
  scripts/stanley.py
  scripts/lidar_cluster.py
  scripts/lattice_planner.py
  scripts/tf_fix.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## include 디렉토리 (현재는 헤더 사용 없음)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## launch 디렉토리 설치
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

