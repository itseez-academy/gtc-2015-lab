#pragma once

#include <vector>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/util.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

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
