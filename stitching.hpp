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
    float registration_time;
    float adjuster_time;
    float matcher_time;
    float find_features_time;
    float blending_time;
    float seam_search_time;
    float composing_time;
    float total_time;
};

class MySphericalWarperGpu: public detail::SphericalWarper
{
public:
    MySphericalWarperGpu(float scale):
    detail::SphericalWarper(scale) {}

    Point warp(const gpu::GpuMat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
                   gpu::CudaMem &dst)
    {
        Rect dst_roi = buildMaps(src.size(), K, R, d_xmap_, d_ymap_);
        dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type(), gpu::CudaMem::ALLOC_ZEROCOPY);
        gpu::GpuMat tmp = dst;
        gpu::remap(src, tmp, d_xmap_, d_ymap_, interp_mode, border_mode);
        return dst_roi.tl();
    }

    Point warp(const gpu::GpuMat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
                                   gpu::GpuMat &dst)
    {
        Rect dst_roi = buildMaps(src.size(), K, R, d_xmap_, d_ymap_);
        dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
        gpu::remap(src, dst, d_xmap_, d_ymap_, interp_mode, border_mode);
        return dst_roi.tl();
    }

    Rect buildMaps(Size src_size, const Mat &K, const Mat &R, gpu::GpuMat &xmap, gpu::GpuMat &ymap)
    {
        projector_.setCameraParams(K, R);

        Point dst_tl, dst_br;
        detectResultRoi(src_size, dst_tl, dst_br);

        gpu::buildWarpSphericalMaps(src_size, Rect(dst_tl, Point(dst_br.x + 1, dst_br.y + 1)),
                                    K, R, projector_.scale, xmap, ymap);

        return Rect(dst_tl, dst_br);
    }

    Rect buildMaps(Size src_size, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap)
    {
        Rect result = buildMaps(src_size, K, R, d_xmap_, d_ymap_);
        d_xmap_.download(xmap);
        d_ymap_.download(ymap);
        return result;
    }

private:
    gpu::GpuMat d_xmap_, d_ymap_, d_src_, d_dst_;
};

void findFeatures(const vector<Mat>& full_imgs, vector<detail::ImageFeatures>& features);
void registerImages(const vector<detail::ImageFeatures>& features,
                    vector<detail::CameraParams>& cameras,
                    Timing& time);
#ifdef USE_GPU
Mat composePano(const vector<Mat>& full_imgs_cpu,
                vector<detail::CameraParams>& cameras,
                float warped_image_scale,
                Timing& time);
#else
Mat composePano(const vector<Mat>& full_imgs,
                vector<detail::CameraParams>& cameras,
                float warped_image_scale,
                Timing& time);
#endif

#endif // STITCHING_HPP
