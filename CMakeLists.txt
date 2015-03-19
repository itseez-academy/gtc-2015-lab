cmake_minimum_required(VERSION 2.8)

project(stitching C CXX)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
add_executable(stitching main.cpp)
target_link_libraries(stitching ${OpenCV_LIBS} ${CUDA_LIBRARIES})
