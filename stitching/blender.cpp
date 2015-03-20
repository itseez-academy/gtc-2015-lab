#include "blender.hpp"

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

void MultiBandBlenderGpu::blend(gpu::GpuMat &d_dst)
{
    for (int i = 0; i <= num_bands_; ++i)
    {
        // d_dst_pyr_laplace_[i] /= d_dst_band_weights_[i];

        gpu::divide(d_dst_pyr_laplace_[i], d_dst_band_weights_[i], d_dst_pyr_laplace_[i], -1, stream_);
    }

    restoreImageFromLaplacePyrGpu(d_dst_pyr_laplace_);

    gpu::GpuMat d_dst_band_weights_roi = d_dst_band_weights_[0](Range(0, dst_roi_final_.height), Range(0, dst_roi_final_.width));
    gpu::GpuMat d_dst_empty_mask;
    gpu::compare(d_dst_band_weights_roi, WEIGHT_EPS, d_dst_empty_mask, cv::CMP_LE, stream_);

    gpu::GpuMat d_dst_pyr_laplace_roi = d_dst_pyr_laplace_[0](Range(0, dst_roi_final_.height), Range(0, dst_roi_final_.width));
    d_dst.create(d_dst_pyr_laplace_roi.size(), CV_16SC3);
    gpu::GpuMat d_dst_pyr_laplace_roi_16u(d_dst_pyr_laplace_roi.size(), CV_16UC4, d_dst_pyr_laplace_roi.data, d_dst_pyr_laplace_roi.step);
    gpu::GpuMat d_dst_16u(d_dst.size(), CV_16UC3, d_dst.data, d_dst.step);
    gpu::cvtColor(d_dst_pyr_laplace_roi_16u, d_dst_16u, cv::COLOR_BGRA2BGR, 0, stream_);
    stream_.enqueueMemSet(d_dst, Scalar::all(0), d_dst_empty_mask);

    stream_.waitForCompletion();
}
