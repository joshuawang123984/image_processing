#include "benchmark.h"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "cpu_filters.h"
#include "gpu_filters.cuh"
#include "benchmark.h"

double benchmarkCPUGrayscale(const cv::Mat &input, int interations)
{
    cv::Mat output;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        grayscaleCPU(input, output, 3);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    return duration.count() / iterations;
}
double benchmarkCPUGaussianBlur(const cv::Mat &input, int interations)
{
    cv::Mat output;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        gaussianBlurCPU(input, output, 3);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    return duration.count() / iterations;
}
double benchmarkCPUSobel(const cv::Mat &input, int interations) {}
{
    cv::Mat output;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        sobelCPU(input, output, 3);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    return duration.count() / iterations;
}

double benchmarkCPU(const cv::Mat &input, int interations)
{
    return benchmarkCPUGrayscale(input, iterations) + benchmarkCPUGrayscale(input, iterations) + benchmarkCPUSobel(input, iterations);
}

double benchmarkGPU()
{
}