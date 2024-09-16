KinectFusion
============

This is an implementation of KinectFusion, based on _Newcombe, Richard A., et al._
**KinectFusion: Real-time dense surface mapping and tracking.**
It makes heavy use of graphics hardware and thus allows for real-time fusion of
depth image scans. Furthermore, exporting of the resulting fused volume is possible either as a pointcloud or a dense surface mesh.

Features
--------
* Real-time fusion of depth scans and corresponding RGB color images
* Easy to use, modern C++14 interface
* Export of the resulting volume as pointcloud
* Export also as dense surface mesh using the MarchingCubes algorithm
* Functions for easy export of pointclouds and meshes into the PLY file format
* Retrieval of calculated camera poses for further processing

Dependencies
------------
* **GCC 5** as higher versions do not work with current nvcc (as of 2017).
* **CUDA 8.0**. In order to provide real-time reconstruction, this library relies on graphics hardware.
Running it exclusively on the CPU is not possible.
* **OpenCV 3.0** or higher. This library heavily depends on the GPU features of OpenCV that have been refactored in the 3.0 release.
Therefore, OpenCV 2 is not supported.
* **Eigen3** for efficient matrix and vector operations.

Prerequisites
-------------
* Adjust CUDA architecture: Set the CUDA architecture version to that of your graphics hardware
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_52,code=sm_52)
Tested with a nVidia GeForce 970, compute capability 5.2, Maxwell architecture
* Set custom opencv path (if necessary):
SET("OpenCV_DIR" "/opt/opencv/usr/local/share/OpenCV")

Usage
-----
```Cpp
#include <iostream>
using namespace std;

#include <kinectfusion.h> // header file from github
using namespace kinectfusion;

#include <Kinect.h>  // Kinect 2 SDK header

#include <opencv2/opencv.hpp>  //opencv header files
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

template<class Interface>
inline void SafeRelease(Interface*& pInterfaceToRelease)  // for safe release, and to free resources 
{
    if (pInterfaceToRelease != nullptr){
        pInterfaceToRelease->Release(); //free resources
        pInterfaceToRelease = nullptr;  // to avoid dangling pointer
    }
}

struct InputFrame { 
    cv::Mat depth_map;  // Depth map (e.g., from IDepthFrame)
    cv::Mat color_map;  // Color map (e.g., from IColorFrame)
};

cv::Mat convertToDepthMap(IDepthFrame* depthFrame) { 
    if (!depthFrame) {
        return cv::Mat();
    }

    // Get depth frame description (width, height)
    IFrameDescription* frameDescription = nullptr; //initialization
    int width = 0;
    int height = 0;
    UINT nBufferSize = 0;
    UINT16* pBuffer = nullptr;

    depthFrame->get_FrameDescription(&frameDescription);
    frameDescription->get_Width(&width);
    frameDescription->get_Height(&height);

    // Create an OpenCV matrix (16-bit unsigned single-channel)
    cv::Mat depthMat(height, width, CV_16U);

    // Access Kinect depth buffer data
    depthFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);
    
    // Copy the depth data to OpenCV Mat
    memcpy(depthMat.data, pBuffer, nBufferSize * sizeof(UINT16));

    // Release frame description
    SafeRelease(frameDescription);

    return depthMat;
}

// Convert IColorFrame to OpenCV Mat
cv::Mat convertToColorMap(IColorFrame* colorFrame) {
    if (!colorFrame) {
        return cv::Mat();
    }

    // Get color frame description
    IFrameDescription* frameDescription = nullptr;
    int width = 0;
    int height = 0;
    ColorImageFormat imageFormat = ColorImageFormat_None;

    colorFrame->get_FrameDescription(&frameDescription);
    frameDescription->get_Width(&width);
    frameDescription->get_Height(&height);
    colorFrame->get_RawColorImageFormat(&imageFormat);

    // Allocate space for the color data (4 channels for RGBA)
    cv::Mat colorMat(height, width, CV_8UC4);

    if (imageFormat == ColorImageFormat_Bgra) {
        colorFrame->CopyConvertedFrameDataToArray(width * height * 4, colorMat.data, ColorImageFormat_Bgra);
    } else {
        colorFrame->CopyRawFrameDataToArray(width * height * 4, colorMat.data);
    }

    // Release frame description
    SafeRelease(frameDescription);

    return colorMat;
}


int main(){
    // Define the Kinect 2 sensor
    IKinectSensor* sensor = nullptr;
    IDepthFrameReader* depthFrameReader = nullptr;
    IColorFrameReader* colorFrameReader = nullptr;

    // Initialize Kinect 2
    HRESULT hr = GetDefaultKinectSensor(&sensor);
    if(FAILED(hr) || !sensor){
        cout << "Failed to initialize Kinect 2 sensor" << endl;
        return -1;
    }

    sensor->Open();

    // Get depth and color readers
    IDepthFrameSource* depthFrameSource = nullptr;
    sensor->get_DepthFrameSource(&depthFrameSource);
    depthFrameSource->OpenReader(&depthFrameReader);

    IColorFrameSource* colorFrameSource = nullptr;
    sensor->get_ColorFrameSource(&colorFrameSource);
    colorFrameSource->OpenReader(&colorFrameReader);

    // Get a global configuration (comes with default values) and adjust some parameters
    kinectfusion::GlobalConfiguration configuration;
    configuration.voxel_scale = 2.f;
    configuration.init_depth = 700.f;
    configuration.distance_threshold = 10.f;
    configuration.angle_threshold = 20.f;

    kinectfusion::CameraParameters camParams;

    // Set the values if necessary (optional since we have default values)
    camParams.image_width = 512;
    camParams.image_height = 424;
    camParams.focal_x = 365.0f;
    camParams.focal_y = 365.0f;
    camParams.principal_x = 256.0f;
    camParams.principal_y = 212.0f;

    // Create a KinectFusion pipeline with Kinect 2 intrinsics
    kinectfusion::Pipeline pipeline { /* Kinect 2 camera intrinsics */ camParams, configuration };

    // Loop over the incoming frames
    bool end = false;
    while (!end) {
        // 1) Grab a frame from the Kinect 2 sensor
        InputFrame frame;

        // Retrieve depth map
        IDepthFrame* depthFrame = nullptr;
        if (SUCCEEDED(depthFrameReader->AcquireLatestFrame(&depthFrame))) {
            // Convert depth frame to your format (e.g., cv::Mat or KinectFusion::DepthMap)
            frame.depth_map = convertToDepthMap(depthFrame);
        }

        // Retrieve color map
        IColorFrame* colorFrame = nullptr;
        if (SUCCEEDED(colorFrameReader->AcquireLatestFrame(&colorFrame))) {
            // Convert color frame to your format (e.g., cv::Mat or KinectFusion::ColorMap)
            frame.color_map = convertToColorMap(colorFrame);
        }

        // 2) Fuse it into the global volume
        bool success = pipeline.process_frame(frame.depth_map, frame.color_map);
        if (!success)
            std::cout << "Frame could not be processed" << std::endl;
    }

    // Retrieve camera poses
    auto poses = pipeline.get_poses();

    // Export surface mesh
    auto mesh = pipeline.extract_mesh();
    kinectfusion::export_ply("data/mesh.ply", mesh);

    // Export pointcloud
    auto pointcloud = pipeline.extract_pointcloud();
    kinectfusion::export_ply("data/pointcloud.ply", pointcloud);

    // Cleanup Kinect 2 resources
    sensor->Close();
    SafeRelease(sensor);
    SafeRelease(depthFrameReader);
    SafeRelease(colorFrameReader);
}

```
For a more in-depth example and implementations of the data sources, have a look at the [KinectFusionApp](https://github.com/chrdiller/KinectFusionApp).

License
-------
This library is licensed under MIT.
