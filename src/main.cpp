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

    // save outputs to jpg
    saveOutput(cpuOutput, "cpu_output.jpg");
    saveOutput(gpuOutput, "gpu_output.jpg");

    // show images
    showImage("Original", input);
    showImage("CPU", cpuOutput);
    showImage("GPU", gpuOutput);

    cv::waitKey(0);

    return 0;
}