#ifndef GPU_FILTERS_CUH
#define GPU_FILTERS_CUH

#include <opencv2/opencv.hpp>

void grayscaleGPU(const cv::Mat &input, cv::Mat &output);

#endif