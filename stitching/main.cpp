#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "stitching.hpp"

#include <cuda_runtime.h>

using namespace std;
using namespace cv;

void help();
int parseCmdArgs(int argc, char** argv);

vector<string> img_names;
string result_name = "result.jpg";

int main(int argc, char* argv[])
{
    //
    // Initialize CUDA context
    //
    cudaFree(NULL);

    //
    // Parse command line options
    //
    int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;

    //
    // Check if have enough images
    //
    size_t num_images = img_names.size();
    if (num_images < 2)
    {
        cout << "Need more images" << endl;
        return -1;
    }

    //
    // Reading input images
    //
    cout << "Reading images..." << endl;
    vector<Mat> full_imgs_cpu(num_images);
#ifndef USE_GPU_COMPOSITION
    for (size_t i = 0; i < num_images; ++i)
    {
        full_imgs_cpu[i] = imread(img_names[i]);
    }
#else
    vector<gpu::GpuMat> full_imgs_gpu(num_images);

    for (size_t i = 0; i < num_images; ++i)
    {
        full_imgs_cpu[i] = imread(img_names[i]);
        full_imgs_gpu[i].upload(full_imgs_cpu[i]);
    }
#endif
    for (size_t i = 0; i < num_images; ++i)
    {
        if (full_imgs_cpu[i].empty())
        {
            cout << "Can't open image " << img_names[i] <<endl;
            return -1;
        }
    }

    Timing time;
    int64 app_start_time = getTickCount();

    //
    // Finding features
    //
    cout << "Finding features..." << endl;
    vector<detail::ImageFeatures> features;
    int64 t = getTickCount();
    findFeatures(full_imgs_cpu, features);
    time.find_features = (getTickCount() - t) / getTickFrequency();

    //
    // Register images
    //
    cout << "Registering images..." << endl;
    vector<detail::CameraParams> cameras;
    t = getTickCount();
    registerImages(features, cameras, time);
    time.registration = (getTickCount() - t) / getTickFrequency();

    //
    // Composition
    //
    cout << "Composing pano..." << endl;
    t = getTickCount();
    float warped_image_scale = FocalLengthMedian(cameras);
#ifdef USE_GPU_COMPOSITION
    Mat result = composePano(full_imgs_gpu, cameras, warped_image_scale, time);
#else
    Mat result = composePano(full_imgs_cpu, cameras, warped_image_scale, time);
#endif
    time.composing = (getTickCount() - t) / getTickFrequency();

    time.total = (getTickCount() - app_start_time) / getTickFrequency();

    imwrite(result_name, result);

    //
    // Reporting performance statistics
    //
    cout << "Done!" << endl << endl;
#ifdef USE_GPU_COMPOSITION
    cout << "Implementation: GPU" << endl;
#else
    cout << "Implementation: CPU" << endl;
#endif
    cout << "Finding features time: "    << time.find_features << " sec" << endl;
    cout << "Images registration time: " << time.registration  << " sec" << endl;
    cout << "   BAdjuster time: "        << time.adjuster      << " sec" << endl;
    cout << "   Matching time: "         << time.matcher       << " sec" << endl;
    cout << "Composing time: "           << time.composing     << " sec" << endl;
    cout << "   Seam search time: "      << time.find_seams    << " sec" << endl;
    cout << "   Blending time: "         << time.blending      << " sec" << endl;
    cout << "Application total time: "   << time.total         << " sec" << endl;

    return 0;
}

void help()
{
    cout <<
        "Image stitching application.\n\n"
        "  $ ./stitching img1 img2 [...imgN]\n\n"
        "Options:\n"
        "  --output <result_img>\n"
        "      The default name is 'result.jpg'.\n";
}

int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        help();
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" ||
            string(argv[i]) == "-h" ||
            string(argv[i]) == "/?")
        {
            help();
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
