#include "utils.h"
#include <iostream>
#include <cuda_runtime.h>

bool isInvalidImage(const cv::Mat &input, bool requiresGrayScale)
{
    if (input.empty())
    {
        std::cerr << "Input image is empty.\n";
        return true;
    }

    if (requiresGrayScale && input.channels() != 1)
    {
        std::cerr << "Expected 1-channel grayscale image.\n";
        return true;
    }

    if (!requiresGrayScale && input.channels() != 3)
    {
        std::cerr << "Expected 3-channel BGR image.\n";
        return true;
    }

    return false;
}

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << ": "
                  << cudaGetErrorString(err)
                  << std::endl;
        exit(1);
    }
}

bool compareImages(const cv::Mat &img1, const cv::Mat &img2)
{
    if (img1.size() != img2.size() || img1.type() != img2.type())
        return false;

    int totalBytes = img1.total() * img1.elemSize();

    for (int i = 0; i < totalBytes; ++i)
    {
        if (img1.data[i] != img2.data[i])
            return false;
    }

    return true;
}

void saveOutput(const cv::Mat &image, const std::string &filename)
{
    cv::imwrite(filename, image);
}

void showImage(const std::string &windowName, const cv::Mat &image)
{
    cv::imshow(windowName, image);
}

void printBenchmark(const std::string &name, double cpuTime, double gpuTime)
{
    std::cout << name << "\n";
    std::cout << "CPU: " << cpuTime << " ms\n";
    std::cout << "GPU: " << gpuTime << " ms\n";
    std::cout << "Speedup: " << cpuTime / gpuTime << "x\n";
}