/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <iostream>
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
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

#if USE_GPU
#include "opencv2/gpu/gpu.hpp"
#endif

using namespace std;
using namespace cv;
using namespace cv::detail;

#if USE_GPU
bool try_gpu = true;
#else
bool try_gpu = true;
#endif

// Default command line args
vector<string> img_names;
double seam_megapix = 0.1;
float conf_thresh = 1.f;
float match_conf = 0.3f;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;
string result_name = "result.jpg";

float find_features_time = 0;
float registration_time = 0;
float adjuster_time = 0;
float matcher_time = 0;
float blending_time = 0;
float seam_search_time = 0;
float composing_time = 0;
float total_time = 0;

void printUsage();
int parseCmdArgs(int argc, char** argv);

void findFeatures(const vector<Mat>& full_imgs, vector<ImageFeatures>& features)
{
    OrbFeaturesFinder finder;

    features.resize(full_imgs.size());

    for (size_t i = 0; i < full_imgs.size(); ++i)
    {
        finder(full_imgs[i], features[i]);
        features[i].img_idx = i;
    }

    finder.collectGarbage();
}

void registerImages(const vector<ImageFeatures>& features, vector<CameraParams>& cameras)
{
    int64 t = getTickCount();
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(try_gpu, match_conf);
    matcher(features, pairwise_matches);
    matcher.collectGarbage();
    matcher_time = (getTickCount() - t) / getTickFrequency();

    HomographyBasedEstimator estimator;
    estimator(features, pairwise_matches, cameras);

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }

    t = getTickCount();
    detail::BundleAdjusterRay adjuster;
    adjuster.setConfThresh(conf_thresh);
    uchar refine_mask_data[] = {1, 1, 1, 0, 1, 1, 0, 0, 0};
    Mat refine_mask(3, 3, CV_8U, refine_mask_data);
    adjuster.setRefinementMask(refine_mask);
    adjuster(features, pairwise_matches, cameras);
    adjuster_time = (getTickCount() - t) / getTickFrequency();

    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R);
    waveCorrect(rmats, wave_correct);
    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].R = rmats[i];
}

void findSeams(Ptr<RotationWarper> full_warper,
               Ptr<RotationWarper> warper,
               const vector<Mat>& full_imgs,
               const vector<CameraParams>& cameras,
               float seam_scale,
               vector<Mat> &images_warped,
               vector<Mat> &masks_warped)
{
    vector<Mat> images(full_imgs.size());
    vector<Mat> images_warped_downscaled(full_imgs.size());
    vector<Mat> masks(full_imgs.size());
    vector<Point> corners(full_imgs.size());

    Ptr<SeamFinder> seam_finder;
#if USE_GPU
        seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR);
#else
        seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
#endif

    for (size_t i = 0; i < full_imgs.size(); ++i)
    {
        resize(full_imgs[i], images[i], Size(), seam_scale, seam_scale);
    }

    // Preapre images masks
    for (size_t i = 0; i < full_imgs.size(); ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp downscaled images and their masks
    for (size_t i = 0; i < images.size(); ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);

        K(0,0) *= (float)seam_scale;
        K(0,2) *= (float)seam_scale;
        K(1,1) *= (float)seam_scale;
        K(1,2) *= (float)seam_scale;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR,
                                  BORDER_REFLECT, images_warped_downscaled[i]);
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    // find seams on downscaled images
    vector<Mat> images_warped_downscaled_f(images.size());
    for (size_t i = 0; i < images_warped.size(); ++i)
    {
        images_warped_downscaled[i].convertTo(images_warped_downscaled_f[i], CV_32F);
    }

    seam_finder->find(images_warped_downscaled_f, corners, masks_warped);

    // upscale to the original resolution
    Mat dilated_mask;
    for (size_t i = 0; i < masks_warped.size(); i++)
    {
        // images
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        full_warper->warp(full_imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);

        // masks
        dilate(masks_warped[i], dilated_mask, Mat());
        resize(dilated_mask, masks_warped[i], images_warped[i].size());
    }
}

#if USE_GPU

class MultiBandBlenderGpu
{
public:
    explicit MultiBandBlenderGpu(int num_bands = 5);

    void prepare(const vector<Point> &corners, const vector<Size> &sizes)
    {
        prepare(resultRoi(corners, sizes));
    }

    void prepare(Rect dst_roi);

    void feed(const Mat &h_img, const Mat &h_mask, Point tl)
    {
        feed(gpu::GpuMat(h_img), gpu::GpuMat(h_mask), tl);
    }

    void blend(Mat &h_dst, Mat &h_dst_mask)
    {
        gpu::GpuMat d_dst, d_dst_mask;
        blend(d_dst, d_dst_mask);
        d_dst.download(h_dst);
        d_dst_mask.download(h_dst_mask);
    }

    void feed(const gpu::GpuMat &d_img, const gpu::GpuMat &d_mask, Point tl);

    void blend(gpu::GpuMat &d_dst, gpu::GpuMat &d_dst_mask);

private:
    void createLaplacePyrGpu(const gpu::GpuMat &d_img, int num_levels, vector<gpu::GpuMat> &d_pyr);
    void restoreImageFromLaplacePyrGpu(vector<gpu::GpuMat> &d_pyr);

    int actual_num_bands_, num_bands_;

    Rect dst_roi_, dst_roi_final_;

    vector<gpu::GpuMat> d_dst_pyr_laplace_;
    vector<gpu::GpuMat> d_dst_band_weights_;

    gpu::Stream stream_;
};

const float WEIGHT_EPS = 1e-5f;

MultiBandBlenderGpu::MultiBandBlenderGpu(int num_bands)
{
    actual_num_bands_ = num_bands;
}

void MultiBandBlenderGpu::prepare(Rect dst_roi)
{
    dst_roi_final_ = dst_roi;

    // Crop unnecessary bands
    double max_len = static_cast<double>(max(dst_roi.width, dst_roi.height));
    num_bands_ = min(actual_num_bands_, static_cast<int>(ceil(log(max_len) / log(2.0))));

    // Add border to the final image, to ensure sizes are divided by (1 << num_bands_)
    dst_roi.width += ((1 << num_bands_) - dst_roi.width % (1 << num_bands_)) % (1 << num_bands_);
    dst_roi.height += ((1 << num_bands_) - dst_roi.height % (1 << num_bands_)) % (1 << num_bands_);
    dst_roi_ = dst_roi;

    d_dst_pyr_laplace_.resize(num_bands_ + 1);
    d_dst_pyr_laplace_[0].create(dst_roi.size(), CV_16SC4);
    d_dst_pyr_laplace_[0].setTo(Scalar::all(0));

    d_dst_band_weights_.resize(num_bands_ + 1);
    d_dst_band_weights_[0].create(dst_roi.size(), CV_32FC1);
    d_dst_band_weights_[0].setTo(WEIGHT_EPS);

    for (int i = 1; i <= num_bands_; ++i)
    {
        d_dst_pyr_laplace_[i].create((d_dst_pyr_laplace_[i - 1].rows + 1) / 2,
                                   (d_dst_pyr_laplace_[i - 1].cols + 1) / 2, CV_16SC4);
        d_dst_band_weights_[i].create((d_dst_band_weights_[i - 1].rows + 1) / 2,
                                    (d_dst_band_weights_[i - 1].cols + 1) / 2, CV_32FC1);

        d_dst_pyr_laplace_[i].setTo(Scalar::all(0));
        d_dst_band_weights_[i].setTo(WEIGHT_EPS);
    }
}

void MultiBandBlenderGpu::createLaplacePyrGpu(const gpu::GpuMat &d_img, int num_levels, vector<gpu::GpuMat> &d_pyr)
{
    d_pyr.resize(num_levels + 1);

    d_pyr[0].create(d_img.size(), CV_16SC4);
    gpu::GpuMat d_img_16u(d_img.size(), CV_16UC3, d_img.data, d_img.step);
    gpu::GpuMat d_pyr0_16u(d_pyr[0].size(), CV_16UC4, d_pyr[0].data, d_pyr[0].step);
    gpu::cvtColor(d_img_16u, d_pyr0_16u, cv::COLOR_BGR2BGRA, 0, stream_);
    for (int i = 0; i < num_levels; ++i)
        gpu::pyrDown(d_pyr[i], d_pyr[i + 1], stream_);

    gpu::GpuMat d_tmp;
    for (int i = 0; i < num_levels; ++i)
    {
        gpu::pyrUp(d_pyr[i + 1], d_tmp, stream_);
        gpu::subtract(d_pyr[i], d_tmp, d_pyr[i], gpu::GpuMat(), -1, stream_);
    }
}

void MultiBandBlenderGpu::feed(const gpu::GpuMat &d_img, const gpu::GpuMat &d_mask, Point tl)
{
    CV_Assert(d_img.type() == CV_16SC3);
    CV_Assert(d_mask.type() == CV_8U);

    // Keep source image in memory with small border
    int gap = 3 * (1 << num_bands_);
    Point tl_new(max(dst_roi_.x, tl.x - gap),
                 max(dst_roi_.y, tl.y - gap));
    Point br_new(min(dst_roi_.br().x, tl.x + d_img.cols + gap),
                 min(dst_roi_.br().y, tl.y + d_img.rows + gap));

    // Ensure coordinates of top-left, bottom-right corners are divided by (1 << num_bands_).
    // After that scale between layers is exactly 2.
    //
    // We do it to avoid interpolation problems when keeping sub-images only. There is no such problem when
    // image is bordered to have size equal to the final image size, but this is too memory hungry approach.
    tl_new.x = dst_roi_.x + (((tl_new.x - dst_roi_.x) >> num_bands_) << num_bands_);
    tl_new.y = dst_roi_.y + (((tl_new.y - dst_roi_.y) >> num_bands_) << num_bands_);
    int width = br_new.x - tl_new.x;
    int height = br_new.y - tl_new.y;
    width += ((1 << num_bands_) - width % (1 << num_bands_)) % (1 << num_bands_);
    height += ((1 << num_bands_) - height % (1 << num_bands_)) % (1 << num_bands_);
    br_new.x = tl_new.x + width;
    br_new.y = tl_new.y + height;
    int dy = max(br_new.y - dst_roi_.br().y, 0);
    int dx = max(br_new.x - dst_roi_.br().x, 0);
    tl_new.x -= dx; br_new.x -= dx;
    tl_new.y -= dy; br_new.y -= dy;

    int top = tl.y - tl_new.y;
    int left = tl.x - tl_new.x;
    int bottom = br_new.y - tl.y - d_img.rows;
    int right = br_new.x - tl.x - d_img.cols;

    // Create the source image Laplacian pyramid
    gpu::GpuMat d_img_with_border;
    gpu::copyMakeBorder(d_img, d_img_with_border, top, bottom, left, right, BORDER_REFLECT, Scalar(), stream_);

    vector<gpu::GpuMat> d_src_pyr_laplace;
    createLaplacePyrGpu(d_img_with_border, num_bands_, d_src_pyr_laplace);

    // Create the weight map Gaussian pyramid
    gpu::GpuMat d_weight_map;
    stream_.enqueueConvert(d_mask, d_weight_map, CV_32F, 1./255.);

    vector<gpu::GpuMat> d_weight_pyr_gauss(num_bands_ + 1);
    gpu::copyMakeBorder(d_weight_map, d_weight_pyr_gauss[0], top, bottom, left, right, BORDER_CONSTANT, Scalar(), stream_);
    for (int i = 0; i < num_bands_; ++i)
        gpu::pyrDown(d_weight_pyr_gauss[i], d_weight_pyr_gauss[i + 1], stream_);

    int y_tl = tl_new.y - dst_roi_.y;
    int y_br = br_new.y - dst_roi_.y;
    int x_tl = tl_new.x - dst_roi_.x;
    int x_br = br_new.x - dst_roi_.x;

    // Add weighted layer of the source image to the final Laplacian pyramid layer
    gpu::GpuMat d_tmp;
    for (int i = 0; i <= num_bands_; ++i)
    {
        gpu::GpuMat d_src_roi = d_src_pyr_laplace[i](cv::Rect(0, 0, x_br-x_tl, y_br-y_tl));
        gpu::GpuMat d_dst_roi = d_dst_pyr_laplace_[i](cv::Rect(x_tl, y_tl, x_br-x_tl, y_br-y_tl));
        gpu::GpuMat d_weight_roi = d_weight_pyr_gauss[i](cv::Rect(0, 0, x_br-x_tl, y_br-y_tl));
        gpu::GpuMat d_dst_weight_roi = d_dst_band_weights_[i](cv::Rect(x_tl, y_tl, x_br-x_tl, y_br-y_tl));

        // dst_roi += src_roi * weight_roi;
        // dst_weight_roi += weight_roi;

        gpu::multiply(d_src_roi, d_weight_roi, d_tmp, 1, -1, stream_);
        gpu::add(d_dst_roi, d_tmp, d_dst_roi, gpu::GpuMat(), -1, stream_);
        gpu::add(d_dst_weight_roi, d_weight_roi, d_dst_weight_roi, gpu::GpuMat(), -1, stream_);

        x_tl /= 2; y_tl /= 2;
        x_br /= 2; y_br /= 2;
    }

    stream_.waitForCompletion();
}

void MultiBandBlenderGpu::restoreImageFromLaplacePyrGpu(vector<gpu::GpuMat> &d_pyr)
{
    if (d_pyr.empty())
        return;

    gpu::GpuMat d_tmp;
    for (size_t i = d_pyr.size() - 1; i > 0; --i)
    {
        gpu::pyrUp(d_pyr[i], d_tmp, stream_);
        gpu::add(d_tmp, d_pyr[i - 1], d_pyr[i - 1], gpu::GpuMat(), -1, stream_);
    }
}

void MultiBandBlenderGpu::blend(gpu::GpuMat &d_dst, gpu::GpuMat &d_dst_mask)
{
    for (int i = 0; i <= num_bands_; ++i)
    {
        // d_dst_pyr_laplace_[i] /= d_dst_band_weights_[i];

        gpu::divide(d_dst_pyr_laplace_[i], d_dst_band_weights_[i], d_dst_pyr_laplace_[i], -1, stream_);
    }

    restoreImageFromLaplacePyrGpu(d_dst_pyr_laplace_);

    gpu::GpuMat d_dst_band_weights_roi = d_dst_band_weights_[0](Range(0, dst_roi_final_.height), Range(0, dst_roi_final_.width));
    gpu::compare(d_dst_band_weights_roi, WEIGHT_EPS, d_dst_mask, cv::CMP_GT, stream_);
    gpu::GpuMat d_dst_mask_inv;
    gpu::bitwise_not(d_dst_mask, d_dst_mask_inv, gpu::GpuMat(), stream_);

    gpu::GpuMat d_dst_pyr_laplace_roi = d_dst_pyr_laplace_[0](Range(0, dst_roi_final_.height), Range(0, dst_roi_final_.width));
    d_dst.create(d_dst_pyr_laplace_roi.size(), CV_16SC3);
    gpu::GpuMat d_dst_pyr_laplace_roi_16u(d_dst_pyr_laplace_roi.size(), CV_16UC4, d_dst_pyr_laplace_roi.data, d_dst_pyr_laplace_roi.step);
    gpu::GpuMat d_dst_16u(d_dst.size(), CV_16UC3, d_dst.data, d_dst.step);
    gpu::cvtColor(d_dst_pyr_laplace_roi_16u, d_dst_16u, cv::COLOR_BGRA2BGR, 0, stream_);
    stream_.enqueueMemSet(d_dst, Scalar::all(0), d_dst_mask_inv);

    stream_.waitForCompletion();
}

#endif

Mat composePano(const vector<Mat>& full_imgs, vector<CameraParams>& cameras, float warped_image_scale)
{
    double seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_imgs[0].size().area()));

    vector<Mat> masks_warped(full_imgs.size());
    vector<Mat> images_warped(full_imgs.size());

    Ptr<WarperCreator> warper_creator;
#if USE_GPU
        warper_creator = new cv::SphericalWarperGpu();
#else
        warper_creator = new cv::SphericalWarper();
#endif

    Ptr<RotationWarper> warper = warper_creator->create(
        static_cast<float>(warped_image_scale * seam_scale));

    Ptr<RotationWarper> full_warper = warper_creator->create(
        static_cast<float>(warped_image_scale));

    int64 t = getTickCount();
    findSeams(full_warper, warper,
              full_imgs, cameras,
              seam_scale,
              images_warped,
              masks_warped);

    seam_search_time = (getTickCount() - t) / getTickFrequency();

    // Update corners and sizes
    t = getTickCount();
    vector<Point> corners(full_imgs.size());
    vector<Size> sizes(full_imgs.size());
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        Rect roi = full_warper->warpRoi(full_imgs[i].size(), K, cameras[i].R);
        corners[i] = roi.tl();
        sizes[i] = roi.size();
    }

    Size dst_sz = resultRoi(corners, sizes).size();
    float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;

#if USE_GPU
    MultiBandBlenderGpu blender(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
    blender.prepare(corners, sizes);

    for (size_t img_idx = 0; img_idx < full_imgs.size(); ++img_idx)
    {
        Mat img_warped_s;
        images_warped[img_idx].convertTo(img_warped_s, CV_16S);

        // Blend the current image
        blender.feed(img_warped_s, masks_warped[img_idx], corners[img_idx]);
    }

    Mat result, result_mask;
    blender.blend(result, result_mask);
#else
    Ptr<Blender> blender = new MultiBandBlender(try_gpu, static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
    blender->prepare(corners, sizes);

    for (size_t img_idx = 0; img_idx < full_imgs.size(); ++img_idx)
    {
        Mat img_warped_s;
        images_warped[img_idx].convertTo(img_warped_s, CV_16S);

        // Blend the current image
        blender->feed(img_warped_s, masks_warped[img_idx], corners[img_idx]);
    }

    Mat result, result_mask;
    blender->blend(result, result_mask);
#endif

    blending_time = (getTickCount() - t) / getTickFrequency();

    return result;
}

int main(int argc, char* argv[])
{
    cv::setBreakOnError(true);

    int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;

    // Check if have enough images
    size_t num_images = img_names.size();
    if (num_images < 2)
    {
        cout << "Need more images" << endl;
        return -1;
    }

    cout << "Reading images..." << endl;
    vector<Mat> full_imgs(num_images);
#if USE_GPU
    vector<gpu::CudaMem> full_imgs_host_mem(num_images);
#endif
    for (size_t i = 0; i < num_images; ++i)
    {
#if USE_GPU
        Mat tmp = imread(img_names[i]);
        full_imgs_host_mem[i].create(tmp.size(), tmp.type());
        full_imgs[i] = full_imgs_host_mem[i];
        tmp.copyTo(full_imgs[i]);
#else
        full_imgs[i] = imread(img_names[i]);
#endif
        if (full_imgs[i].empty())
        {
            cout << "Can't open image " << img_names[i] <<endl;
            return -1;
        }
    }

    int64 app_start_time = getTickCount();

    cout << "Finding features..." << endl;
    vector<ImageFeatures> features;
    int64 t = getTickCount();
    findFeatures(full_imgs, features);
    find_features_time = (getTickCount() - t) / getTickFrequency();

    cout << "Registering images..." << endl;
    vector<CameraParams> cameras;
    t = getTickCount();
    registerImages(features, cameras);
    registration_time = (getTickCount() - t) / getTickFrequency();

    // Find median focal length
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
        focals.push_back(cameras[i].focal);

    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] +
                             focals[focals.size() / 2]) * 0.5f;

    cout << "Composing pano..." << endl;
    t = getTickCount();
    Mat result = composePano(full_imgs, cameras, warped_image_scale);
    composing_time = (getTickCount() - t) / getTickFrequency();

    total_time = (getTickCount() - app_start_time) / getTickFrequency();

    imwrite(result_name, result);

    cout << endl;
    cout << "Finding features time: " << find_features_time << " sec" << endl;
    cout << "Images registration time: " << registration_time << " sec"<< endl;
    cout << "   Adjuster time: " << adjuster_time << " sec" << endl;
    cout << "   Matching time: " << matcher_time << " sec" << endl;
    cout << "Composing time: " << composing_time << " sec" << endl;
    cout << "   Seam search time: " << seam_search_time << " sec" << endl;
    cout << "   Blending time: " << blending_time << " sec" << endl;
    cout << "Application total time: " << total_time << " sec" << endl;

    return 0;
}

void printUsage()
{
    cout <<
        "Rotation model images stitcher.\n\n"
        "stitching img1 img2 [...imgN]\n\n"
        "Flags:\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n";
}


int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage();
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage();
            return -1;
        }
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else
        {
            img_names.push_back(argv[i]);
        }
    }
    return 0;
}
