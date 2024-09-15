#include <iostream>
using namespace std;

#include <kinectfusion.h>

#include <Kinect.h>  // Kinect 2 SDK header

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

template<class Interface>
inline void SafeRelease(Interface*& pInterfaceToRelease)
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
    IFrameDescription* frameDescription = nullptr;
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

    // Create a KinectFusion pipeline with Kinect 2 intrinsics
    kinectfusion::Pipeline pipeline { /* Kinect 2 camera intrinsics */ camera_parameters, configuration };

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
