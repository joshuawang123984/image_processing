#ifndef GPU_FILTERS_CUH
#define GPU_FILTERS_CUH

#include <opencv2/opencv.hpp>

void grayscaleGPU(const cv::Mat &input, cv::Mat &output);
void gaussianBlurGPU(const cv::Mat &input, cv::Mat &output, int kernelSize);
void sobelGPU(const cv::Mat &input, cv::Mat &output);

#endif