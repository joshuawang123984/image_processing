#include "benchmark.h"
#include <chrono>
#include <opencv2/opencv.hpp>

#include "cpu_filters.h"
#include "gpu_filters.cuh"

double benchmarkCPU(const cv::Mat &input, int iterations, void (*filter)(const cv::Mat &, cv::Mat &))

{
    cv::Mat output;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        filter(input, output);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    return duration.count() / iterations;
}

double benchmarkGPU(const cv::Mat &input, int iterations, void (*filter)(const cv::Mat &, cv::Mat &))
{
    cv::Mat output;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        filter(input, output);
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    return duration.count() / iterations;
}