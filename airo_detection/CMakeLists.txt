cmake_minimum_required(VERSION 2.8.3)
project(airo_detection)

find_package(catkin REQUIRED COMPONENTS 
             rospy
             cv_bridge
             sensor_msgs)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS cv_bridge rospy sensor_msgs roscpp
)

#############
## Install ##
#############
catkin_install_python(PROGRAMS 
  scripts/threeDim_fruit_generator.py
  scripts/dbscan.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/groovy/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
