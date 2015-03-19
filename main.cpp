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
#include <cuda_runtime.h>
#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/device/common.hpp"

#include "stitching.hpp"

using namespace std;
using namespace cv;

void printUsage();
int parseCmdArgs(int argc, char** argv);

vector<string> img_names;
string result_name = "result.jpg";

int main(int argc, char* argv[])
{
    cudaSafeCall(cudaSetDeviceFlags(cudaDeviceMapHost));
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
    vector<Mat> full_imgs_cpu(num_images);
#if USE_GPU
    vector<gpu::CudaMem> full_imgs_host_mem(num_images);
    vector<gpu::GpuMat> full_imgs_gpu(num_images);
#endif
    for (size_t i = 0; i < num_images; ++i)
    {
#if USE_GPU
        Mat tmp = imread(img_names[i]);
        full_imgs_host_mem[i].create(tmp.size(), tmp.type());
        full_imgs_cpu[i] = full_imgs_host_mem[i];
        tmp.copyTo(full_imgs_cpu[i]);
        full_imgs_gpu[i].upload(full_imgs_cpu[i]);
#else
        full_imgs_cpu[i] = imread(img_names[i]);
#endif
        if (full_imgs_cpu[i].empty())
        {
            cout << "Can't open image " << img_names[i] <<endl;
            return -1;
        }
    }

    Timing time;
    int64 app_start_time = getTickCount();

    cout << "Finding features..." << endl;
    vector<detail::ImageFeatures> features;
    int64 t = getTickCount();
    findFeatures(full_imgs_cpu, features);
    time.find_features = (getTickCount() - t) / getTickFrequency();

    cout << "Registering images..." << endl;
    vector<detail::CameraParams> cameras;
    t = getTickCount();
    registerImages(features, cameras, time);
    time.registration = (getTickCount() - t) / getTickFrequency();

    // Find median focal length
    float warped_image_scale = FocalLengthMedian(cameras);

    cout << "Composing pano..." << endl;
    t = getTickCount();
#ifdef USE_GPU
    Mat result = composePano(full_imgs_gpu, cameras, warped_image_scale, time);
#else
    Mat result = composePano(full_imgs_cpu, cameras, warped_image_scale, time);
#endif
    time.composing = (getTickCount() - t) / getTickFrequency();

    time.total = (getTickCount() - app_start_time) / getTickFrequency();

    imwrite(result_name, result);

    cout << "Done!" << endl << endl;
    cout << "Finding features time: " << time.find_features << " sec" << endl;
    cout << "Images registration time: " << time.registration << " sec"<< endl;
    cout << "   Adjuster time: " << time.adjuster << " sec" << endl;
    cout << "   Matching time: " << time.matcher << " sec" << endl;
    cout << "Composing time: " << time.composing << " sec" << endl;
    cout << "   Seam search time: " << time.find_seams << " sec" << endl;
    cout << "   Blending time: " << time.blending << " sec" << endl;
    cout << "Application total time: " << time.total << " sec" << endl;

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
