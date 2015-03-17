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

using namespace std;
using namespace cv;
using namespace cv::detail;

// Default command line args
vector<string> img_names;
bool try_gpu = true;
double work_megapix = 0.6;
double seam_megapix = 0.1;
float conf_thresh = 1.f;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;
string result_name = "result.jpg";

float find_features_time = 0;
float partwise_matching_time = 0;
float warping_time = 0;
float compositing_time = 0;
float total_time = 0;

void printUsage();
int parseCmdArgs(int argc, char** argv);

void findFeatures(const vector<Mat>& full_imgs, vector<ImageFeatures>& features, double work_scale)
{
    Mat img;
    OrbFeaturesFinder finder;

    features.resize(full_imgs.size());

    for (size_t i = 0; i < full_imgs.size(); ++i)
    {
        resize(full_imgs[i], img, Size(), work_scale, work_scale);

        finder(img, features[i]);
        features[i].img_idx = i;
        cout << "Features in image #" << i+1 << ": " << features[i].keypoints.size() << endl;
    }

    finder.collectGarbage();
}

float registerImages(const vector<ImageFeatures>& features, vector<CameraParams>& cameras)
{
    cout << "Pairwise matching..." << endl;
    double t = getTickCount();
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(try_gpu, match_conf);
    matcher(features, pairwise_matches);
    matcher.collectGarbage();
    partwise_matching_time = (getTickCount() - t) / getTickFrequency();

    cout << "Image registration..." << endl;
    HomographyBasedEstimator estimator;
    estimator(features, pairwise_matches, cameras);

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }

    detail::BundleAdjusterRay adjuster;
    adjuster.setConfThresh(conf_thresh);
    uchar refine_mask_data[] = {1, 1, 1, 0, 1, 1, 0, 0, 0};
    Mat refine_mask(3, 3, CV_8U, refine_mask_data);
    adjuster.setRefinementMask(refine_mask);
    adjuster(features, pairwise_matches, cameras);
}

Mat composePano(const vector<Mat>& full_imgs, vector<CameraParams>& cameras, double work_scale, float warped_image_scale)
{
    double seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_imgs[0].size().area()));
    double seam_work_aspect = seam_scale / work_scale;
    double compose_work_aspect = 1. / work_scale;

    vector<Mat> images(full_imgs.size());
    vector<Mat> masks(full_imgs.size());
    vector<Point> corners(full_imgs.size());
    vector<Mat> masks_warped(full_imgs.size());
    vector<Mat> images_warped(full_imgs.size());
    vector<Size> sizes(full_imgs.size());

    cout << "Downscaling for futher processing..." << endl;
    for (size_t i = 0; i < full_imgs.size(); ++i)
        resize(full_imgs[i], images[i], Size(), seam_scale, seam_scale);

    // Preapre images masks
    for (size_t i = 0; i < full_imgs.size(); ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks

    cout << "Warping images (auxiliary)..." << endl;
    double t = getTickCount();
    Ptr<WarperCreator> warper_creator;
#if defined(HAVE_OPENCV_GPU)
    if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
        warper_creator = new cv::SphericalWarperGpu();
    else
#endif
        warper_creator = new cv::SphericalWarper();

    Ptr<RotationWarper> warper = warper_creator->create(
                static_cast<float>(warped_image_scale * seam_work_aspect));

    for (size_t i = 0; i < full_imgs.size(); ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa; K(1,1) *= swa; K(1,2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR,
                                  BORDER_REFLECT, images_warped[i]);

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<Mat> images_warped_f(full_imgs.size());
    for (size_t i = 0; i < images_warped.size(); ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    warping_time = (getTickCount() - t) / getTickFrequency();

    Ptr<SeamFinder> seam_finder;
#if defined(HAVE_OPENCV_GPU)
    if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
        seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR);
    else
#endif
        seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);

    seam_finder->find(images_warped_f, corners, masks_warped);

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    cout << "Compositing..." << endl;
    t = getTickCount();

    // Update warped image scale
    warped_image_scale *= static_cast<float>(compose_work_aspect);
    warper = warper_creator->create(warped_image_scale);

    // Update corners and sizes
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        // Update intrinsics
        cameras[i].focal *= compose_work_aspect;
        cameras[i].ppx *= compose_work_aspect;
        cameras[i].ppy *= compose_work_aspect;

        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        Rect roi = warper->warpRoi(full_imgs[i].size(), K, cameras[i].R);
        corners[i] = roi.tl();
        sizes[i] = roi.size();
    }

    Size dst_sz = resultRoi(corners, sizes).size();
    float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
    Ptr<Blender> blender = new MultiBandBlender(try_gpu, static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
    blender->prepare(corners, sizes);

    for (size_t img_idx = 0; img_idx < full_imgs.size(); ++img_idx)
    {
        cout << "Compositing image #" << img_idx << endl;

        Mat img_warped, img_warped_s;
        Mat dilated_mask, seam_mask, mask, mask_warped;

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(full_imgs[img_idx], K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(full_imgs[img_idx].size(), CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;

        // Blend the current image
        blender->feed(img_warped_s, mask_warped, corners[img_idx]);
    }

    Mat result, result_mask;
    blender->blend(result, result_mask);

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

    vector<Mat> full_imgs(num_images);
    for (size_t i = 0; i < num_images; ++i)
    {
        full_imgs[i] = imread(img_names[i]);
        if (full_imgs[i].empty())
        {
            cout << "Can't open image " << img_names[i] <<endl;
            return -1;
        }
    }

    cout << "Images reading finished" << endl;

    int64 app_start_time = getTickCount();
    double work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_imgs[0].size().area()));

    cout << "Finding features..." << endl;
    vector<ImageFeatures> features;
    int64 t = getTickCount();
    findFeatures(full_imgs, features, work_scale);
    find_features_time = (getTickCount() - t) / getTickFrequency();

    vector<CameraParams> cameras;
    registerImages(features, cameras);

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

    t = getTickCount();
    Mat result = composePano(full_imgs, cameras, work_scale, warped_image_scale);
    compositing_time = (getTickCount() - t) / getTickFrequency();

    total_time = (getTickCount() - app_start_time) / getTickFrequency();

    imwrite(result_name, result);

    cout << "Finding features, time: " << find_features_time << " sec" << endl;
    cout << "Pairwise matching, time: " << partwise_matching_time << " sec"<< endl;
    cout << "Warping images, time: " << warping_time << " sec" << endl;
    cout << "Compositing, time: " << compositing_time << " sec" << endl;
    cout << "Finished, total time: " << total_time << " sec" << endl;

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
