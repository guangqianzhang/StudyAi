cmake_minimum_required(VERSION 3.0.2)
project(matchproject)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -fopenmp")

set(ALL_TARGET_LIBRARIES "")
# include(../../cmake/glog.cmake)
# include(../../cmake/YAML.cmake)
include(../src/cmake/OPENCV.cmake)

file(GLOB ALL_SRCS "*.cpp")


add_executable(${PROJECT_NAME}_node matchproject.cpp )

target_link_libraries(${PROJECT_NAME}_node
 
  ${ALL_TARGET_LIBRARIES}
)