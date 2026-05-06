#pragma once

#include <opencv2/opencv.hpp>
#include <string>
// #include <cuda_runtime.h>

bool isInvalidImage(const cv::Mat &input, bool requiresGrayScale);
// void checkCudaError(cudaError_t err, const char *msg);
bool compareImages(const cv::Mat &img1, const cv::Mat &img2);
void saveOutput(const cv::Mat &image, const std::string &filename);
void showImage(const std::string &windowName, const cv::Mat &image);
void printBenchmark(const std::string &name, double cpuTime, double gpuTime);