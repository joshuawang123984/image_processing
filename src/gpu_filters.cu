#include "gpu_filters.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/opencv.hpp>
#include <iostream>

__global__ void grayscaleKernel(const unsigned char *input, unsigned char *output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = (y * width + x) * channels;

    unsigned char b = input[idx + 0];
    unsigned char g = input[idx + 1];
    unsigned char r = input[idx + 2];

    unsigned char gray = static_cast<unsigned char>(0.114f * b + 0.587f * g + 0.299f * r);
    output[y * width + x] = gray;
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