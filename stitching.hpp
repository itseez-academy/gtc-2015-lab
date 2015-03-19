#ifndef STITCHING_HPP
#define STITCHING_HPP

#include "opencv2/gpu/gpu.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/camera.hpp"

//
// Use these parameters to configure pipeline
//
const bool USE_GPU_MATCHING = true;
#define USE_GPU_COMPOSITION

struct Timing
{
    float registration;
    float adjuster;
    float matcher;
    float find_features;
    float blending;
    float find_seams;
    float composing;
    float total;
};

void findFeatures(const std::vector<cv::Mat>& imgs,
                  std::vector<cv::detail::ImageFeatures>& features);

void registerImages(const std::vector<cv::detail::ImageFeatures>& features,
                    std::vector<cv::detail::CameraParams>& cameras,
                    Timing& time);

#ifdef USE_GPU_COMPOSITION
cv::Mat composePano(const std::vector<cv::gpu::GpuMat>& imgs,
                std::vector<cv::detail::CameraParams>& cameras,
                float warped_image_scale,
                Timing& time);
#else
cv::Mat composePano(const std::vector<cv::Mat>& imgs,
                std::vector<cv::detail::CameraParams>& cameras,
                float warped_image_scale,
                Timing& time);
#endif

float FocalLengthMedian(std::vector<cv::detail::CameraParams>& cameras);

#endif // STITCHING_HPP
