#include "gpu_filters.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/opencv.hpp>
#include <iostream>

#define TILE 16
__constant__ int Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

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

            int curr_idx = ((y + i) * width + (x + j)) * channels;
            // gaussian blur
            float sigma = kernelSize / 6.0f;
            float coeff = std::exp(-(i * i + j * j) / (2 * sigma * sigma));

            blueSum += input[curr_idx + 0] * coeff;
            greenSum += pixel[curr_idx + 1] * coeff;
            redSum += pixel[curr_idx + 2] * coeff;

            totalWeight += coeff;
        }
    }

    output[idx + 0] = (unsigned char)(blueSum / totalWeight);
    output[idx + 1] = (unsigned char)(greenSum / totalWeight);
    output[idx + 2] = (unsigned char)(redSum / totalWeight);
}

__global__ void sobelKernel(const unsigned char *input, unsigned char *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    int sumX = 0;
    int sumY = 0;

    // for 3 x 3 kernels
    for (int i = -1; i < 2; ++i)
    {
        for (int j = -1; j < 2; ++j)
        {

            if (y + i < 0 || y + i >= height || x + j < 0 || x + j >= width)
                continue;

            int curr_idx = (y + i) * width + (x + j);

            sumX += input[curr_idx] * Gx[(i + 1) * 3 + (j + 1)];
            sumY += input[curr_idx] * Gy[(i + 1) * 3 + (j + 1)];
        }
    }

    int magnitude = (int)sqrtf((float)(sumX * sumX + sumY * sumY));
    magnitude = magnitude > 255 ? 255 : magnitude;
    output[idx] = (unsigned char)magnitude;
}

void grayscaleGPU(const cv::Mat &input, cv::Mat &output)
{
    // boiler plate for allocating to GPU mem
    unsigned char *d_input;
    unsigned char *d_output;

    int inputSize = input.rows * input.cols * input.channels();
    int outputSize = input.rows * input.cols;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_input, input.data, inputSize, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((input.cols + 15) / 16, (input.rows + 15) / 16);
    grayscaleKernel<<<blocks, threads>>>(d_input, d_output, input.cols, input.rows, input.channels);

    cudaMemcpy(gpuOutput, d_output, outputSize, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

void gaussianBlurGPU(const cv::Mat &input, cv::Mat &output, int kernelSize)
{
}

void sobelGPU(const cv::Mat &input, cv::Mat &output) {}