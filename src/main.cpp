#include <iostream>
#include <opencv2/opencv.hpp>

#include "cpu_filters.h"
#include "gpu_filters.cuh"
#include "benchmark.h"
#include "utils.h"

int main()
{
    cv::Mat input = cv::imread("input.jpg", cv::IMREAD_COLOR);

    if (input.empty())
    {
        std::cerr << "Image failed to load.\n";
        return -1;
    }

    cv::Mat cpuOutput, gpuOutput;

    grayscaleCPU(input, cpuOutput);
    grayscaleGPU(input, gpuOutput);

    // save outputs to jpg
    saveOutput(cpuOutput, "cpu_output.jpg");
    saveOutput(cpuOutput, "gpu_output.jpg");

    // show images
    showImage("Original", input);
    showImage("CPU", cpuOutput);
    showImage("GPU", gpuOutput);

    cv::waitKey(0);

    return 0;
}