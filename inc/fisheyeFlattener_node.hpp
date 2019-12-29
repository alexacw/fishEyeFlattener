#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <camera_model/camera_models/CameraFactory.h>
#include "cv_bridge/cv_bridge.h"
#include <filesystem>

std::vector<cv::Mat> generateAllUndistMap(camera_model::CameraPtr p_cam,
                                          Eigen::Vector3d rotation,
                                          const unsigned &imgWidth,
                                          const double &fov //degree
);

cv::Mat genOneUndistMap(
    camera_model::CameraPtr p_cam,
    Eigen::AngleAxis<double> rotation,
    const unsigned &imgWidth,
    const unsigned &imgHeight,
    const double &f);