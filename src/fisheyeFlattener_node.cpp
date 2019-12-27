#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <camera_model/camera_models/CameraFactory.h>
#include "cv_bridge/cv_bridge.h"
#include <filesystem>

#define DEG_TO_RAD (M_PI * 2.0 / 360.0)

namespace globalVar
{
std::string inputTopic;
std::string outputTopicPrefix;
std::string camFilePath;
camera_model::CameraPtr cam;
Eigen::Vector3d cameraRotation;
int imgWidth = 0;
double fov0 = 0;
double fov1 = 0;
std::array<cv::Mat, 5> undistMaps;
ros::Publisher img_pub[5];
} // namespace globalVar

std::array<cv::Mat, 5> generateUndistMap(
    camera_model::CameraPtr p_cam,
    Eigen::Vector3d rotation,
    const unsigned &imgWidth,
    const double &fov //degree
);

void imgCB(const sensor_msgs::Image::ConstPtr &msg);

int main(int argc, char **argv)
{
    using namespace globalVar;

    ros::init(argc, argv, "fisheyeFlattener_node");
    ros::NodeHandle nh("~");

    // obtain camera intrinsics
    nh.param<std::string>("cam_file", camFilePath, "cam.yaml");
    ROS_INFO(camFilePath.c_str());
    if (!std::filesystem::exists(camFilePath))
    {
        ROS_ERROR("Camera file does not exist.");
        return 1;
    }
    cam = camera_model::CameraFactory::instance()
              ->generateCameraFromYamlFile(camFilePath);

    // remapping parameters
    nh.param<double>("rotationVectorX", cameraRotation.x(), 0);
    nh.param<double>("rotationVectorY", cameraRotation.y(), 0);
    nh.param<double>("rotationVectorZ", cameraRotation.z(), 0);
    nh.param<double>("fov0", fov0, 90);
    nh.param<double>("fov1", fov1, 90);
    nh.param<int>("imgWidth", imgWidth, 500);
    if (imgWidth <= 0)
    {
        ROS_ERROR("Resolution must be non-negative");
        return 1;
    }
    if (fov0 <= 0 || fov1 <= 0)
    {
        ROS_ERROR("FOV must be non-negative");
        return 1;
    }

    nh.param<std::string>("inputTopic", inputTopic, "img");
    nh.param<std::string>("outputTopicPrefix", outputTopicPrefix, "flatImg");

    undistMaps = generateUndistMap(cam, cameraRotation, 100, 270);

    for (int i = 0; i < 5; i++)
    {
        img_pub[i] = nh.advertise<sensor_msgs::Image>(outputTopicPrefix + "_" + std::to_string(i), 3);
    }
    ros::Subscriber img_sub = nh.subscribe(inputTopic, 3, imgCB);
    ros::spin();
    return 0;
}

void imgCB(const sensor_msgs::Image::ConstPtr &msg){

};

std::array<cv::Mat, 5> generateUndistMap(
    camera_model::CameraPtr p_cam,
    Eigen::Vector3d rotation,
    const unsigned &imgWidth,
    const double &fov //degree
)
{
    Eigen::Vector2d temp;
    p_cam->spaceToPlane(rotation, temp);
    std::cout << "rotation on img plane:\n"
              << temp << std::endl;

    double sideVerticalFOV = (fov - 180) * DEG_TO_RAD;
    double centerFOV = M_PI * 2 - fov;
    int sideImgHeight = sideVerticalFOV / centerFOV * imgWidth;
    std::array<cv::Mat, 5> maps;
    maps[0] = cv::Mat(imgWidth, imgWidth, CV_32FC2);
    for (int i = 1; i < 5; i++)
        maps[1] = cv::Mat(imgWidth, sideImgHeight, CV_32FC2);

    // rotation applied for all points
    Eigen::AngleAxis t = Eigen::AngleAxis<double>(rotation.norm(), rotation.normalized()).inverse();

    // calculate focal length of fake pinhole cameras
    double f = imgWidth / 2 / tan(fov / 2);

    //center
    for (int x = 0; x < imgWidth; x++)
        for (int y = 0; y < imgWidth; y++)
        {
            Eigen::Vector3d objPoint =
                t * Eigen::Vector3d((x - (double)imgWidth / 2) / f,
                                    (y - (double)imgWidth / 2) / f,
                                    1);
            Eigen::Vector2d imgPoint;
            p_cam->spaceToPlane(objPoint, imgPoint);
            maps[0].at<cv::Vec2f>(cv::Point(x, y)) = cv::Vec2f(imgPoint.x(), imgPoint.y());
        }

    std::cout << "Center range:\n"
              << maps[0].at<cv::Vec2f>(cv::Point(0, 0)) << std::endl
              << maps[0].at<cv::Vec2f>(cv::Point(imgWidth - 1, imgWidth - 1)) << std::endl;

    //x direction
    for (int x = 0; x < imgWidth; x++)
        for (int y = 0; y < sideImgHeight; y++)
        {
            Eigen::Vector3d objPoint =
                t * Eigen::Vector3d(1,
                                    ((double)x - (double)imgWidth / 2) / f,
                                    ((double)y - (double)sideImgHeight / 2) / f);
            Eigen::Vector2d imgPoint;
            p_cam->spaceToPlane(objPoint, imgPoint);
            maps[1].at<cv::Vec2f>(cv::Point(x, y)) = cv::Vec2f(imgPoint.x(), imgPoint.y());
        }

    return maps;
}
