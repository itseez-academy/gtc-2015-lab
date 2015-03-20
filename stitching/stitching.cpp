#include "stitching.hpp"

#ifdef USE_GPU_COMPOSITION
#include "blender.hpp"
#endif

#include "opencv2/stitching/stitcher.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace std;
using namespace cv;

double seam_megapix = 0.1;
float conf_thresh = 1.f;
float match_conf = 0.3f;
detail::WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
int blend_type = detail::Blender::MULTI_BAND;
float blend_strength = 5;


void findFeatures(const vector<Mat>& imgs, vector<detail::ImageFeatures>& features)
{
    detail::OrbFeaturesFinder finder;

    features.resize(imgs.size());
    for (size_t i = 0; i < imgs.size(); ++i)
    {
        finder(imgs[i], features[i]);
        features[i].img_idx = i;
    }

    finder.collectGarbage();
}


void registerImages(const vector<detail::ImageFeatures>& features,
                    vector<detail::CameraParams>& cameras, Timing& time)
{
    vector<detail::MatchesInfo> pairwise_matches;
    detail::BestOf2NearestMatcher matcher(USE_GPU_MATCHING, match_conf);
    detail::BundleAdjusterRay adjuster;
    adjuster.setConfThresh(conf_thresh);
    uchar refine_mask_data[] = {1, 1, 1, 0, 1, 1, 0, 0, 0};
    Mat refine_mask(3, 3, CV_8U, refine_mask_data);

    // feature matching
    int64 t = getTickCount();
    matcher(features, pairwise_matches);
    time.matcher = (getTickCount() - t) / getTickFrequency();

    matcher.collectGarbage();

    detail::HomographyBasedEstimator estimator;
    estimator(features, pairwise_matches, cameras);

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }

    // bundle adjustment
    t = getTickCount();
    adjuster.setRefinementMask(refine_mask);
    adjuster(features, pairwise_matches, cameras);
    time.adjuster = (getTickCount() - t) / getTickFrequency();

    // horizon correction
    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R);
    waveCorrect(rmats, wave_correct);
    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].R = rmats[i];
}


#ifdef USE_GPU_COMPOSITION
void findSeams(detail::SphericalWarperGpu& warper_full,
               detail::SphericalWarperGpu& warper_downscaled,
               const vector<gpu::GpuMat>& imgs,
               const vector<detail::CameraParams>& cameras,
               float seam_scale,
               vector<gpu::GpuMat>& images_warped,
               vector<gpu::GpuMat>& masks_warped)
{
    vector<gpu::GpuMat> images_downscaled(imgs.size());
    vector<gpu::GpuMat> images_downscaled_warped(imgs.size());
    vector<gpu::GpuMat> masks(imgs.size());
    vector<gpu::GpuMat> masks_warped_downscaled(imgs.size());
    vector<Point> corners(imgs.size());

    detail::VoronoiSeamFinder seam_finder;

    for (size_t i = 0; i < imgs.size(); ++i)
        gpu::resize(imgs[i], images_downscaled[i], Size(), seam_scale, seam_scale);

    // Preapre images masks
    for (size_t i = 0; i < imgs.size(); ++i)
    {
        masks[i].create(images_downscaled[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp downscaled images and their masks
    Mat_<float> K;
    for (size_t i = 0; i < images_downscaled.size(); ++i)
    {
        cameras[i].K().convertTo(K, CV_32F);

        K(0,0) *= (float)seam_scale;
        K(0,2) *= (float)seam_scale;
        K(1,1) *= (float)seam_scale;
        K(1,2) *= (float)seam_scale;

        corners[i] = warper_downscaled.warp(images_downscaled[i], K, cameras[i].R,
                                            INTER_LINEAR, BORDER_REFLECT,
                                            images_downscaled_warped[i]);
        warper_downscaled.warp(masks[i], K, cameras[i].R, INTER_NEAREST,
                               BORDER_CONSTANT, masks_warped_downscaled[i]);
    }

    vector<Mat> masks_warped_cpu(imgs.size());
    for (size_t i = 0; i < imgs.size(); i++)
        masks_warped_downscaled[i].download(masks_warped_cpu[i]);

    vector<Size> sizes(images_downscaled_warped.size());
    for (size_t i = 0; i < images_downscaled_warped.size(); ++i)
        sizes[i] = images_downscaled_warped[i].size();

    seam_finder.find(sizes, corners, masks_warped_cpu);

    for (size_t i = 0; i < masks_warped_downscaled.size(); i++)
        masks_warped_downscaled[i].upload(masks_warped_cpu[i]);

    // upscale to the original resolution
    gpu::GpuMat dilated_mask;
    for (size_t i = 0; i < masks_warped_downscaled.size(); i++)
    {
        // images - warp as is
        cameras[i].K().convertTo(K, CV_32F);
        warper_full.warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);

        // masks - upscale after seaming
        gpu::dilate(masks_warped_downscaled[i], dilated_mask, Mat());
        gpu::resize(dilated_mask, masks_warped[i], images_warped[i].size());
    }
}
#else
void findSeams(detail::SphericalWarper& warper_full,
               detail::SphericalWarper& warper_downscaled,
               const vector<Mat>& imgs,
               const vector<detail::CameraParams>& cameras,
               float seam_scale,
               vector<Mat> &images_warped,
               vector<Mat> &masks_warped)
{
    vector<Mat> images_downscaled(imgs.size());
    vector<Mat> images_downscaled_warped(imgs.size());
    vector<Mat> masks(imgs.size());
    vector<Point> corners(imgs.size());

    detail::VoronoiSeamFinder seam_finder;

    for (size_t i = 0; i < imgs.size(); ++i)
        resize(imgs[i], images_downscaled[i], Size(), seam_scale, seam_scale);

    // Preapre images masks
    for (size_t i = 0; i < imgs.size(); ++i)
    {
        masks[i].create(images_downscaled[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp downscaled images and their masks
    Mat_<float> K;
    for (size_t i = 0; i < images_downscaled.size(); ++i)
    {
        cameras[i].K().convertTo(K, CV_32F);

        K(0,0) *= (float)seam_scale;
        K(0,2) *= (float)seam_scale;
        K(1,1) *= (float)seam_scale;
        K(1,2) *= (float)seam_scale;

        corners[i] = warper_downscaled.warp(images_downscaled[i], K, cameras[i].R,
                                            INTER_LINEAR, BORDER_REFLECT,
                                            images_downscaled_warped[i]);
        warper_downscaled.warp(masks[i], K, cameras[i].R, INTER_NEAREST,
                               BORDER_CONSTANT, masks_warped[i]);
    }

    vector<Size> sizes(images_downscaled_warped.size());
    for (size_t i = 0; i < images_downscaled_warped.size(); ++i)
        sizes[i] = images_downscaled_warped[i].size();

    seam_finder.find(sizes, corners, masks_warped);

    // upscale to the original resolution
    Mat dilated_mask;
    for (size_t i = 0; i < masks_warped.size(); i++)
    {
        // images - warp as is
        cameras[i].K().convertTo(K, CV_32F);
        warper_full.warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);

        // masks - upscale after seaming
        dilate(masks_warped[i], dilated_mask, Mat());
        resize(dilated_mask, masks_warped[i], images_warped[i].size());
    }
}
#endif


#ifdef USE_GPU_COMPOSITION
Mat composePano(const vector<gpu::GpuMat>& imgs,
                vector<detail::CameraParams>& cameras,
                float warped_image_scale,
                Timing& time)
{
    double seam_scale = min(1.0, sqrt(seam_megapix * 1e6 /
                                      imgs[0].size().area()));

    vector<gpu::GpuMat> masks_warped(imgs.size());
    vector<gpu::GpuMat> images_warped(imgs.size());
    vector<Point> corners(imgs.size());
    vector<Size> sizes(imgs.size());

    detail::SphericalWarperGpu warper_full(
                static_cast<float>(warped_image_scale));
    detail::SphericalWarperGpu warper_downscaled(
                static_cast<float>(warped_image_scale * seam_scale));

    int64 t = getTickCount();
    findSeams(warper_full, warper_downscaled, imgs, cameras,
              seam_scale, images_warped, masks_warped);
    time.find_seams = (getTickCount() - t) / getTickFrequency();

    // Update corners and sizes
    t = getTickCount();
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        Rect roi = warper_full.warpRoi(imgs[i].size(), K, cameras[i].R);
        corners[i] = roi.tl();
        sizes[i] = roi.size();
    }

    Size result_size = detail::resultRoi(corners, sizes).size();
    float blend_width = sqrt(static_cast<float>(result_size.area())) *
                             blend_strength / 100.f;
    MultiBandBlenderGpu blender(
                static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
    blender.prepare(detail::resultRoi(corners, sizes));

    gpu::GpuMat img_warped_s;
    for (size_t i = 0; i < imgs.size(); ++i)
    {
        images_warped[i].convertTo(img_warped_s, CV_16S);
        blender.feed(img_warped_s, masks_warped[i], corners[i]);
    }

    Mat result;
    blender.blend(result);

    time.blending = (getTickCount() - t) / getTickFrequency();

    return result;
}
#else
Mat composePano(const vector<Mat>& imgs,
                vector<detail::CameraParams>& cameras,
                float warped_image_scale,
                Timing& time)
{
    double seam_scale = min(1.0, sqrt(seam_megapix * 1e6 /
                                      imgs[0].size().area()));

    vector<Mat> masks_warped(imgs.size());
    vector<Mat> images_warped(imgs.size());
    vector<Point> corners(imgs.size());
    vector<Size> sizes(imgs.size());

    detail::SphericalWarper warper_full(
                static_cast<float>(warped_image_scale));
    detail::SphericalWarper warper_downscaled(
                static_cast<float>(warped_image_scale * seam_scale));

    int64 t = getTickCount();
    findSeams(warper_full, warper_downscaled, imgs, cameras,
              seam_scale, images_warped, masks_warped);

    time.find_seams = (getTickCount() - t) / getTickFrequency();

    // Update corners and sizes
    t = getTickCount();
    Mat K;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        cameras[i].K().convertTo(K, CV_32F);
        Rect roi = warper_full.warpRoi(imgs[i].size(), K, cameras[i].R);
        corners[i] = roi.tl();
        sizes[i] = roi.size();
    }

    Size result_size = detail::resultRoi(corners, sizes).size();
    float blend_width = sqrt(static_cast<float>(result_size.area())) *
                        blend_strength / 100.f;
    detail::MultiBandBlender blender(
                static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));

    blender.prepare(detail::resultRoi(corners, sizes));

    Mat img_warped_s;
    for (size_t img_idx = 0; img_idx < imgs.size(); ++img_idx)
    {
        images_warped[img_idx].convertTo(img_warped_s, CV_16S);
        blender.feed(img_warped_s, masks_warped[img_idx], corners[img_idx]);
    }

    Mat result, result_mask;
    blender.blend(result, result_mask);
    time.blending = (getTickCount() - t) / getTickFrequency();

    return result;
}
#endif

float FocalLengthMedian(vector<detail::CameraParams>& cameras)
{
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
        focals.push_back(cameras[i].focal);

    sort(focals.begin(), focals.end());

    float median;
    if (focals.size() % 2 == 1)
        median = static_cast<float>(focals[focals.size() / 2]);
    else
        median = static_cast<float>(focals[focals.size() / 2 - 1] +
                             focals[focals.size() / 2]) * 0.5f;

    return median;
}
