#include "cpu_filters.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <cmath>

void grayscaleCPU(const cv::Mat &input, cv::Mat &output)
{
    if (isInvalidImage(input, false))
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
    if (isInvalidImage(input, false))
    {
        return;
    }

    if (kernelSize % 2 == 0 || kernelSize < 1)
    {
        std::cerr << "Kernel size must be odd and positive.\n";
        return;
    }

    output = cv::Mat(input.rows, input.cols, CV_8UC3);
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
            output.at<cv::Vec3b>(r, c) = blurPixel;
        }
    }
}

void sobelCPU(const cv::Mat &input, cv::Mat &output)
{
    if (isInvalidImage(input, true))
    {
        return;
    }

    output = cv::Mat(input.rows, input.cols, CV_8UC1, cv::Scalar(0));
    // 3x3 kernels
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int r = 1; r < input.rows - 1; ++r)
    {
        for (int c = 1; c < input.cols - 1; ++c)
        {
            int sumX = 0;
            int sumY = 0;

            // for 3 x 3 kernels
            for (int i = -1; i < 2; ++i)
            {
                for (int j = -1; j < 2; ++j)
                {
                    unsigned char pixel = input.at<unsigned char>(r + i, c + j);
                    sumX += pixel * Gx[i + 1][j + 1];
                    sumY += pixel * Gy[i + 1][j + 1];
                }
            }

            int magnitude = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
            magnitude = std::min(255, magnitude);

            output.at<unsigned char>(r, c) = static_cast<unsigned char>(magnitude);
        }
    }
}