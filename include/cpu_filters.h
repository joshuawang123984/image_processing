#ifndef CPU_FILTERS_H
#define CPU_FILTERS_H

#include <opencv2/opencv.hpp>

void grayscaleCPU(const cv::Mat &input, cv::Mat &output);
void gaussianBlurCPU(const cv::Mat &input, cv::Mat &output, int filters);
void sobelCPU(const cv::Mat &input, cv::Mat &output);

#endif