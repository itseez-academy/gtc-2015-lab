#include "stitching.hpp"

#ifdef USE_GPU
#include "blender.hpp"
#endif

using namespace std;
using namespace cv;

double seam_megapix = 0.1;
float conf_thresh = 1.f;
float match_conf = 0.3f;
detail::WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
int blend_type = detail::Blender::MULTI_BAND;
float blend_strength = 5;

void findFeatures(const vector<Mat>& full_imgs, vector<detail::ImageFeatures>& features)
{
    detail::OrbFeaturesFinder finder;

    features.resize(full_imgs.size());

    for (size_t i = 0; i < full_imgs.size(); ++i)
    {
        finder(full_imgs[i], features[i]);
        features[i].img_idx = i;
    }

    finder.collectGarbage();
}

void registerImages(const vector<detail::ImageFeatures>& features, vector<detail::CameraParams>& cameras, Timing& time)
{
    int64 t = getTickCount();
    vector<detail::MatchesInfo> pairwise_matches;
    detail::BestOf2NearestMatcher matcher(try_gpu, match_conf);
    matcher(features, pairwise_matches);
    matcher.collectGarbage();
    time.matcher_time = (getTickCount() - t) / getTickFrequency();

    detail::HomographyBasedEstimator estimator;
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
    time.adjuster_time = (getTickCount() - t) / getTickFrequency();

    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R);
    waveCorrect(rmats, wave_correct);
    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].R = rmats[i];
}

#ifdef USE_GPU
void findSeams(detail::SphericalWarperGpu& full_warper,
               MySphericalWarperGpu& warper,
               const vector<gpu::GpuMat>& full_imgs,
               const vector<detail::CameraParams>& cameras,
               float seam_scale,
               vector<gpu::GpuMat>& images_warped,
               vector<gpu::GpuMat>& masks_warped)
{
    vector<gpu::GpuMat> images(full_imgs.size());
    vector<gpu::GpuMat> images_warped_downscaled(full_imgs.size());
    vector<gpu::GpuMat> masks(full_imgs.size());
    vector<Point> corners(full_imgs.size());
    vector<gpu::CudaMem> masks_warped_cumem(full_imgs.size());

    detail::VoronoiSeamFinder seam_finder;

    for (size_t i = 0; i < full_imgs.size(); ++i)
        gpu::resize(full_imgs[i], images[i], Size(), seam_scale, seam_scale);

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

        corners[i] = warper.warp(images[i], K, cameras[i].R, INTER_LINEAR,
                                  BORDER_REFLECT, images_warped_downscaled[i]);
        warper.warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped_cumem[i]);
    }

    vector<Mat> masks_warped_cpu(masks_warped_cumem.size());
    vector<gpu::GpuMat> masks_warped_small(masks_warped_cumem.size());
    for (size_t i = 0; i < masks_warped_cumem.size(); i++)
    {
        masks_warped_cpu[i] = masks_warped_cumem[i];
        masks_warped_small[i] = masks_warped_cumem[i];
    }

    vector<Size> sizes_(images_warped_downscaled.size());
    for (size_t i = 0; i < images_warped_downscaled.size(); ++i)
        sizes_[i] = images_warped_downscaled[i].size();

    seam_finder.find(sizes_, corners, masks_warped_cpu);

    // upscale to the original resolution
    gpu::GpuMat dilated_mask;
    for (size_t i = 0; i < masks_warped_small.size(); i++)
    {
        // images
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        full_warper.warp(full_imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);

        // masks
        gpu::dilate(masks_warped_small[i], dilated_mask, Mat());
        gpu::resize(dilated_mask, masks_warped[i], images_warped[i].size());
    }
}
#else
void findSeams(detail::SphericalWarper& full_warper,
               detail::SphericalWarper& warper,
               const vector<Mat>& full_imgs,
               const vector<detail::CameraParams>& cameras,
               float seam_scale,
               vector<Mat> &images_warped,
               vector<Mat> &masks_warped)
{
    vector<Mat> images(full_imgs.size());
    vector<Mat> images_warped_downscaled(full_imgs.size());
    vector<Mat> masks(full_imgs.size());
    vector<Point> corners(full_imgs.size());

    Ptr<detail::SeamFinder> seam_finder = new detail::VoronoiSeamFinder();

    for (size_t i = 0; i < full_imgs.size(); ++i)
        resize(full_imgs[i], images[i], Size(), seam_scale, seam_scale);

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

        corners[i] = warper.warp(images[i], K, cameras[i].R, INTER_LINEAR,
                                  BORDER_REFLECT, images_warped_downscaled[i]);
        warper.warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    seam_finder->find(images_warped_downscaled, corners, masks_warped);

    // upscale to the original resolution
    Mat dilated_mask;
    for (size_t i = 0; i < masks_warped.size(); i++)
    {
        // images
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        full_warper.warp(full_imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);

        // masks
        dilate(masks_warped[i], dilated_mask, Mat());
        resize(dilated_mask, masks_warped[i], images_warped[i].size());
    }
}
#endif

#ifdef USE_GPU
Mat composePano(const vector<gpu::GpuMat>& full_imgs,
                vector<detail::CameraParams>& cameras,
                float warped_image_scale,
                Timing& time)
{
    double seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_imgs[0].size().area()));

    vector<gpu::GpuMat> masks_warped(full_imgs.size());
    vector<gpu::GpuMat> images_warped(full_imgs.size());

    MySphericalWarperGpu warper(static_cast<float>(warped_image_scale * seam_scale));
    detail::SphericalWarperGpu full_warper(static_cast<float>(warped_image_scale));

    int64 t = getTickCount();
    findSeams(full_warper, warper,
              full_imgs, cameras,
              seam_scale,
              images_warped,
              masks_warped);

    time.seam_search_time = (getTickCount() - t) / getTickFrequency();

    // Update corners and sizes
    t = getTickCount();
    vector<Point> corners(full_imgs.size());
    vector<Size> sizes(full_imgs.size());
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        Rect roi = full_warper.warpRoi(full_imgs[i].size(), K, cameras[i].R);
        corners[i] = roi.tl();
        sizes[i] = roi.size();
    }

    Size dst_sz = detail::resultRoi(corners, sizes).size();
    float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
    MultiBandBlenderGpu blender(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
    blender.prepare(detail::resultRoi(corners, sizes));

    for (size_t img_idx = 0; img_idx < full_imgs.size(); ++img_idx)
    {
        gpu::GpuMat img_warped_s;
        images_warped[img_idx].convertTo(img_warped_s, CV_16S);
        blender.feed(img_warped_s, masks_warped[img_idx], corners[img_idx]);
    }

    Mat result, result_mask;
    blender.blend(result, result_mask);

    time.blending_time = (getTickCount() - t) / getTickFrequency();

    return result;
}
#else
Mat composePano(const vector<Mat>& full_imgs,
                vector<detail::CameraParams>& cameras,
                float warped_image_scale,
                Timing& time)
{
    double seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_imgs[0].size().area()));

    vector<Mat> masks_warped(full_imgs.size());
    vector<Mat> images_warped(full_imgs.size());

    detail::SphericalWarper warper(static_cast<float>(warped_image_scale * seam_scale));
    detail::SphericalWarper full_warper(static_cast<float>(warped_image_scale));

    int64 t = getTickCount();
    findSeams(full_warper, warper,
              full_imgs, cameras,
              seam_scale,
              images_warped,
              masks_warped);

    time.seam_search_time = (getTickCount() - t) / getTickFrequency();

    // Update corners and sizes
    t = getTickCount();
    vector<Point> corners(full_imgs.size());
    vector<Size> sizes(full_imgs.size());
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        Rect roi = full_warper.warpRoi(full_imgs[i].size(), K, cameras[i].R);
        corners[i] = roi.tl();
        sizes[i] = roi.size();
    }

    Size dst_sz = detail::resultRoi(corners, sizes).size();
    float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
    Ptr<detail::Blender> blender = new detail::MultiBandBlender(try_gpu, static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
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

    time.blending_time = (getTickCount() - t) / getTickFrequency();

    return result;
}
#endif
