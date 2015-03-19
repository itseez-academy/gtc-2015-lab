#ifndef STITCHING_HPP
#define STITCHING_HPP

#include "opencv2/gpu/gpu.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#define USE_GPU 1

using namespace std;
using namespace cv;

#ifdef USE_GPU
const bool try_gpu = true;
#else
const bool try_gpu = true;
#endif

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

void findFeatures(const vector<Mat>& full_imgs,
                  vector<detail::ImageFeatures>& features);

void registerImages(const vector<detail::ImageFeatures>& features,
                    vector<detail::CameraParams>& cameras,
                    Timing& time);

#ifdef USE_GPU
Mat composePano(const vector<gpu::GpuMat>& imgs,
                vector<detail::CameraParams>& cameras,
                float warped_image_scale,
                Timing& time);
#else
Mat composePano(const vector<Mat>& imgs,
                vector<detail::CameraParams>& cameras,
                float warped_image_scale,
                Timing& time);
#endif

float FocalLengthMedian(vector<detail::CameraParams>& cameras);

#endif // STITCHING_HPP
