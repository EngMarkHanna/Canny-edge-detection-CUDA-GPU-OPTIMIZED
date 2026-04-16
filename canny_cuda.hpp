#pragma once
#include <opencv2/core.hpp>

namespace cv
{
    bool cuda_canny_sobel_mag_dir_nms(const Mat& src, Mat& dst, float low_thresh, float high_thresh, int aperture_size, bool L2gradient);
}

bool cuda_Canny(cv::InputArray _src, cv::OutputArray _dst, float low_thresh, float high_thresh, int aperture_size, bool L2gradient);