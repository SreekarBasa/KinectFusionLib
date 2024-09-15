cmake_minimum_required(VERSION 3.5)
project(KinectFusionLib)

# Use modern C++
set(CMAKE_CXX_STANDARD 20)

# Setting output paths
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(KINECT_SDK_DIR "C:/Program Files/Microsoft SDKs/Kinect/v2.0_1409")
include_directories(${KINECT_SDK_DIR}/inc)

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

## CUDA
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    include_directories("${CUDAToolkit_INCLUDE_DIRS}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -std=c++11 -gencode arch=compute_52,code=sm_52")
endif()

## OpenCV
set(OpenCV_DIR "C:/Program Files/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found. Make sure OpenCV is correctly installed and OpenCV_DIR is set.")
endif()

# Manually set the path to Eigen's include directory
set(Eigen3_INCLUDE_DIR "C:/Program Files (x86)/eigen-3.4.0")
include_directories(${Eigen3_INCLUDE_DIR})
if(NOT EXISTS "${Eigen3_INCLUDE_DIR}/Eigen/Dense")
    message(FATAL_ERROR "Eigen3 not found at ${Eigen3_INCLUDE_DIR}")
endif()

# ------------------------------------------------
# CMake flags
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")

# Project files
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

include_directories("${PROJECT_INCLUDE_DIR}" "${PROJECT_SOURCE_DIR}/cuda/include")
file(GLOB KinectFusion_SRCS "${PROJECT_SOURCE_DIR}/cuda/*.cu" "${PROJECT_SOURCE_DIR}/*.cpp")

# Build library
add_library(KinectFusion STATIC ${KinectFusion_SRCS})

# Link libraries
target_link_libraries(KinectFusion ${OpenCV_LIBS} ${CUDAToolkit_LIBRARIES} "${KINECT_SDK_DIR}/Lib/*.lib")

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS_PATH}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARIES: ${CUDAToolkit_LIBRARIES}")
message(STATUS "Eigen3_INCLUDE_DIR: ${Eigen3_INCLUDE_DIR}")
