cmake_minimum_required(VERSION 2.8)

project(stitching C CXX)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

if(NOT CMAKE_CROSSCOMPILING)
    get_filename_component(cuda_libs_path ${CUDA_CUDART_LIBRARY} PATH)
    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${cuda_libs_path}" isSystemDir)
    if("${isSystemDir}" STREQUAL "-1")
        set(CMAKE_INSTALL_RPATH "${cuda_libs_path}")
    endif()
endif()

add_executable(stitching main.cpp stitching.cpp blender.cpp)
target_link_libraries(stitching ${OpenCV_LIBS} ${CUDA_LIBRARIES})
