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
double compose_megapix = -1;
float conf_thresh = 1.f;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;
string result_name = "result.jpg";

void printUsage();
int parseCmdArgs(int argc, char** argv);

int main(int argc, char* argv[])
{
    float find_features_time = 0;
    float partwise_matching_time = 0;
    float warping_time = 0;
    float compositing_time = 0;
    float total_time = 0;

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
    vector<Size> full_img_sizes(num_images);
    for (size_t i = 0; i < num_images; ++i)
    {
        full_imgs[i] = imread(img_names[i]);
        full_img_sizes[i] = full_imgs[i].size();

        if (full_imgs[i].empty())
        {
            cout << "Can't open image " << img_names[i] <<endl;
            return -1;
        }
    }

    cout << "Files reading finished" << endl;

    double compose_scale = 1;
    bool is_compose_scale_set = false;

    int64 app_start_time = getTickCount();

    cout << "Finding features..." << endl;
    int64 t = getTickCount();

    OrbFeaturesFinder finder;

    Mat img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);

    double work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img_sizes[0].area()));
    double seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img_sizes[0].area()));
    double seam_work_aspect = seam_scale / work_scale;

    for (int i = 0; i < num_images; ++i)
    {
        resize(full_imgs[i], img, Size(), work_scale, work_scale);

        finder(img, features[i]);
        features[i].img_idx = i;
        cout << "Features in image #" << i+1 << ": " << features[i].keypoints.size() << endl;

        resize(full_imgs[i], img, Size(), seam_scale, seam_scale);
        images[i] = img.clone();
    }

    finder.collectGarbage();
    img.release();

    find_features_time = (getTickCount() - t) / getTickFrequency();

    cout << "Pairwise matching..."<< endl;
    t = getTickCount();
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(try_gpu, match_conf);
    matcher(features, pairwise_matches);
    matcher.collectGarbage();

    partwise_matching_time = (getTickCount() - t) / getTickFrequency();

    // Leave only images we are sure are from the same panorama
    cout << "Finding biggest connected component..." << endl;
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<string> img_names_subset;
    vector<Mat> full_imgs_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_imgs_subset.push_back(full_imgs[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    images = img_subset;
    img_names = img_names_subset;
    full_imgs = full_imgs_subset;
    full_img_sizes = full_img_sizes_subset;

    // Check if we still have enough images
    num_images = img_names.size();
    if (num_images < 2)
    {
        cout << "Need more images" << endl;
        return -1;
    }

    cout << "Image registration..." << endl;
    HomographyBasedEstimator estimator;
    vector<CameraParams> cameras;
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

    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R);
    waveCorrect(rmats, wave_correct);
    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].R = rmats[i];

    cout << "Warping images (auxiliary)..." << endl;
    t = getTickCount();

    vector<Point> corners(num_images);
    vector<Mat> masks_warped(num_images);
    vector<Mat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<Mat> masks(num_images);

    // Preapre images masks
    for (size_t i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks

    Ptr<WarperCreator> warper_creator;
#if defined(HAVE_OPENCV_GPU)
    if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
    {
        warper_creator = new cv::SphericalWarperGpu();
    }
    else
#endif
    {
        warper_creator = new cv::SphericalWarper();
    }

    Ptr<RotationWarper> warper = warper_creator->create(
                static_cast<float>(warped_image_scale * seam_work_aspect));

    for (size_t i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR,
                                  BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<Mat> images_warped_f(num_images);
    for (size_t i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    warping_time = (getTickCount() - t) / getTickFrequency();

    cout << "Exposure compensation..." << endl;
    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    compensator->feed(corners, images_warped, masks_warped);

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

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    double compose_work_aspect = 1;

    for (size_t img_idx = 0; img_idx < num_images; ++img_idx)
    {
        cout << "Compositing image #" << indices[img_idx]+1 << endl;

        // Read image and resize it if necessary
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img_sizes[img_idx].area()));
            is_compose_scale_set = true;

            // Compute relative scales
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (size_t i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_imgs[img_idx], img, Size(), compose_scale, compose_scale);
        else
            img = full_imgs[img_idx];
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // Compensate exposure
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;

        if (blender.empty())
        {
            blender = Blender::createDefault(blend_type, try_gpu);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_gpu);
            else
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
            }
            blender->prepare(corners, sizes);
        }

        // Blend the current image
        blender->feed(img_warped_s, mask_warped, corners[img_idx]);
    }

    Mat result, result_mask;
    blender->blend(result, result_mask);

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
