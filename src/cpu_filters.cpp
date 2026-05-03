#include "cpu_filters.h"
#include <opencv2/opencv.hpp>
#include <cmath>

bool isEmpty(const cv::Mat &input)
{
    if (input.empty())
    {
        std::cerr << "Input image is empty.\n";
        return true;
    }

    if (input.channels() != 3)
    {
        std::cerr << "Expected 3-channel BGR image.\n";
        return true;
    }

    return false;
}

void grayscaleCPU(const cv::Mat &input, cv::Mat &output)
{
    if (isEmpty(input))
    {
        return;
    }

    output = cv::Mat(input.rows, input.cols, CV_8UC1);

    for (int r = 0; r < input.rows; ++r)
    {
        for (int c = 0; c < input.cols; ++c)
        {
            cv::Vec3b pixel = input.at<cv::Vec3b>(r, c);

            unsigned char gray = static_cast<unsigned char>(0.114f * pixel[0] + 0.587f * pixel[1] + 0.299f * pixel[2]);
            output.at<unsigned char>(r, c) = gray;
        }
    }
}

void gaussianBlurCPU(const cv::Mat &input, cv::Mat &output, int kernelSize)
{
    if (isEmpty(input))
    {
        return;
    }

    if (kernelSize % 2 == 0 || kernelSize < 1)
    {
        std::cerr << "Kernel size must be odd and positive.\n";
        return;
    }

    output = cv::Mat(input.rows, input.cols, CV_8UC1);
    int radius = kernelSize / 2;

    for (int r = 0; r < input.rows; ++r)
    {
        for (int c = 0; c < input.cols; ++c)
        {
            float blueSum = 0.0f;
            float greenSum = 0.0f;
            float redSum = 0.0f;
            float totalWeight = 0.0f;

            for (int i = -radius; i < radius + 1; ++i)
            {
                if (r + i < 0)
                {
                    continue;
                }

                if (r + i >= input.rows)
                {
                    break;
                }

                for (int j = -radius; j < radius + 1; ++j)
                {
                    if (c + j < 0)
                    {
                        continue;
                    }

                    if (c + j >= input.cols)
                    {
                        break;
                    }

                    cv::Vec3b pixel = input.at<cv::Vec3b>(r + i, c + j);
                    // not really a gaussian formula but whatever
                    float coeff = 1.0f / (std::abs(i) * std::abs(j) + 1);

                    blueSum += pixel[0] * coeff;
                    greenSum += pixel[1] * coeff;
                    redSum += pixel[2] * coeff;

                    totalWeight += coeff;
                }
            }

            cv::Vec3b blurPixel;

            blurPixel[0] = static_cast<unsigned char>(blueSum / totalWeight);
            blurPixel[1] = static_cast<unsigned char>(greenSum / totalWeight);
            blurPixel[2] = static_cast<unsigned char>(redSum / totalWeight);
            unsigned char blur = {blueSum, greenSum, redSum};
            output.at<cv::Vec3b>(r, c) = blurPixel;
        }
    }
}

void sobelCPU(const cv::Mat &input, cv::Mat &output)
{
    if (isEmpty(input))
    {
        return;
    }
}