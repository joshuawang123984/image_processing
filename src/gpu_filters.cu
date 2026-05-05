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
__global__ void gaussianBlurKernel(const unsigned char *input, unsigned char *output, int width, int height, int channels, int kernelSize)
{
    __shared__ float data[][];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = (y * width + x) * channels;
    int radius = kernelSize / 2;
    float sigma = kernelSize / 6.0f;

    float blueSum = 0.0f;
    float greenSum = 0.0f;
    float redSum = 0.0f;
    float totalWeight = 0.0f;

    for (int i = -radius; i < radius + 1; ++i)
    {
        if (y + i < 0)
        {
            continue;
        }

        if (y + i >= input.rows)
        {
            break;
        }

        for (int j = -radius; j < radius + 1; ++j)
        {
            if (x + j < 0)
            {
                continue;
            }

            if (x + j >= input.cols)
            {
                break;
            }

            int idx = ((y + i) * width + (x + j)) * channels;
            // gaussian blur
            float sigma = kernelSize / 6.0f;
            float coeff = std::exp(-(i * i + j * j) / (2 * sigma * sigma));

            blueSum += input[idx + 0] * coeff;
            greenSum += pixel[idx + 1] * coeff;
            redSum += pixel[idx + 2] * coeff;

            totalWeight += coeff;
        }
    }

    int outIdx = (y * width + x) * channels);

    output[outIdx + 0] = (unsigned char)(blueSum / totalWeight);
    output[outIdx + 1] = (unsigned char)(greenSum / totalWeight);
    output[outIdx + 2] = (unsigned char)(redSum / totalWeight);
}
__global__ void sobelKernel(...)
{
}

void grayscaleGPU(const cv::Mat &input, cv::Mat &output)
{
}