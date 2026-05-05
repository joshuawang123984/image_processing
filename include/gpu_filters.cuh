#ifndef GPU_FILTERS_CUH
#define GPU_FILTERS_CUH

#include <opencv2/opencv.hpp>

void grayscaleKernel(const unsigned char *input, unsigned char *output, int width, int height, int channels);
void gaussianBlurKernel(const unsigned char *input, unsigned char *output, int width, int height, int channels, int kernelSize);
void sobelKernel(const unsigned char *input, unsigned char *output, int width, int height, int channels);

void grayscaleGPU(const cv::Mat &input, cv::Mat &output);
void gaussianBlurGPU(const cv::Mat &input, cv::Mat &output, int kernelSize);
void sobelGPU(const cv::Mat &input, cv::Mat &output);
#endif