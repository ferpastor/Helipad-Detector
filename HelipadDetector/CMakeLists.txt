cmake_minimum_required(VERSION 2.8.3)

set(PROJECT_NAME HelipadDetector)
project(${PROJECT_NAME})

### Use version 2011 of C++ (c++11). By default ROS uses c++98
#see: http://stackoverflow.com/questions/10851247/how-to-activate-c-11-in-cmake
#see: http://stackoverflow.com/questions/10984442/how-to-detect-c11-support-of-a-compiler-with-cmake
add_definitions(-std=c++11)
#add_definitions(-std=c++0x)
#add_definitions(-std=c++03)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries

 


#Helipad Detector
set(HelipadDetector_SOURCE_DIR
	src/source) 
	
set(HelipadDetector_INCLUDE_DIR
	src/include)

set(HelipadDetector_SOURCE_FILES
	${HelipadDetector_SOURCE_DIR}/HelipadDetector.cpp
)
 
set(HelipadDetector_HEADER_FILES
	${HelipadDetector_INCLUDE_DIR}/HelipadDetector.h
)

find_package(catkin REQUIRED
                COMPONENTS roscpp image_transport cv_bridge droneMsgsROS droneModuleROS referenceFrames)

find_package(OpenCV REQUIRED)



catkin_package(
        DEPENDS OpenCV
        CATKIN_DEPENDS roscpp image_transport cv_bridge droneMsgsROS droneModuleROS referenceFrames
  )


include_directories(${HelipadDetector_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})



add_library(HelipadDetector_lib ${HelipadDetector_SOURCE_FILES} ${HelipadDetector_HEADER_FILES})
add_dependencies(HelipadDetector_lib ${catkin_EXPORTED_TARGETS})
target_link_libraries(HelipadDetector_lib ${OpenCV_LIBS})
target_link_libraries(HelipadDetector_lib ${catkin_LIBRARIES})


add_executable(HelipadDetector ${HelipadDetector_SOURCE_DIR}/HelipadDetectorNode.cpp)
add_dependencies(HelipadDetector ${catkin_EXPORTED_TARGETS})
target_link_libraries(HelipadDetector HelipadDetector_lib)
target_link_libraries(HelipadDetector ${OpenCV_LIBS})
target_link_libraries(HelipadDetector ${catkin_LIBRARIES})

