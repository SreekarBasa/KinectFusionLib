cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0146 NEW)  # Set the policy for CUDA
project(KinectFusionLib)

# Use modern C++
set(CMAKE_CXX_STANDARD 20)

# Setting output paths
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# ------------------------------------------------
#                 Dependencies
# ------------------------------------------------
## CUDA

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")
find_package(CUDAToolkit REQUIRED)
IF (CUDAToolkit_FOUND)
    include_directories("${CUDAToolkit_INCLUDE_DIRS}")
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_52,code=sm_52)
ENDIF ()

## OpenCV
# Update OpenCV version to 4.5 or another version that you have installed
set(OpenCV_DIR "C:/Users/91901/Desktop/BTP/kinect/opencv/build")  # Adjust this path
find_package(OpenCV 4.5 REQUIRED)  # Use 4.5 instead of 4.10

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_DIR}/lib)
else()
    message(FATAL_ERROR "OpenCV not found. Make sure OpenCV is correctly installed and OpenCV_DIR is set.")
endif()

## Eigen3
find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
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
target_link_libraries(KinectFusion ${OpenCV_LIBS} ${CUDA_CUDART_LIB})
