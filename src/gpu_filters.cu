#include "gpu_filters.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/opencv.hpp>
#include <iostream>

__global__ void grayscaleKernel(...)
{
}
__global__ void gaussianBlurKernel(...)
{
}
__global__ void sobelKernel(...)
{
}

void grayscaleGPU(const cv::Mat &input, cv::Mat &output)
{
}