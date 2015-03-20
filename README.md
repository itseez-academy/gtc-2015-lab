# Introduction

`GTC2015_stitching` project is a simple Panorama Stitching pipeline
that demonstrates various CPU and GPU optimizations possible on the
Jetson TK1 platform.

The original implementation is written in C++ for Tegra CPU. We will
measure its performance and then try to improve it using the CUDA
technology. We will see that even the NEON-optimized code is less
efficient than GPU implementation. However, memory transfers are quite
expensive, so it is not enough to just replace bottlenecks and better
to port the entire pipeline to the GPU.

# Prerequisites

For Unix: SSH to the Jetson TK1 board (user: ubuntu, password: ubuntu):

    $ ssh ubuntu@<board address>

For Windows:
 1. Go to www.putty.org, click "You can download PuTTY here",
   download putty.zip (NOT .exe, but .zip)
 2. Extract putty.zip, open the extracted directory and run putty.exe
 3. Enter board address and press "Open"
 4. Enter user name and password (user: ubuntu, password: ubuntu).

For ALL OS: open yet another terminal on desktop for files sharing.
Use Putty tools folder on Windows to use `pscp` tool without
environment changes.

Get the source code and input images from GitHub:

    $ wget https://github.com/Itseez/gtc-2015-lab/archive/master.zip
    $ unzip master.zip

Turn off power management on Tegra in order to get reliable
performance measurements:

    $ cd ./gtc-2015-lab-master
    $ sudo ./freqScale_l4t.sh

# Build & Run

Execute the following commands on your Jetson TK1 board:

    $ make -j3
    $ ./stitching ../images/*.jpg

You will see the performance report in your console. You can also
check the final panorama exported to the `result.jpg` file.

# Working on the lab

By default the application works on CPU. And it has three major
bottleneck functions:

  - Seam finding `cv::detail::VoronoiSeamFinder::find`
  - Composition `cv::detail::SphericalWarper::warp` and
    `detail::MultiBandBlender::blend`

Source code contains time measurements that allow to detect
bottlenecks and all potential optimization steps. We will try to
switch from CPU to GPU version using CUDA and then tune it for the
Jetson TK1 platform.

OpenCV provides GPU-based implementations for Warping and
Blending steps that give speedup ranging from 1.5x to 2.5x.
`VoronoiSeamFinder::find` method does not have GPU implementation, but
switch to the GPU pipeline allows to significantly improve performance
of the seam search procedure in general.

## Detailed instructions

### Step 1. Profile CPU implementation

Run the application and record performance data.

### Step 2. Switch to GPU pipeline

In file stitching/stitching.hpp:11 uncomment #define USE_GPU_COMPOSITION
line. It switches the pipeline to GPU version.

Major changes:
 - `cv::Mat` => `cv::gpu::GpuMat`
 - Blending on GPU
 - Warping on GPU
 - Utility image processing steps on GPU

# Conclusion

 - OpenCV for Tegra helps to prototype and optimize computational pipelines
 - GPU greatly helps in real-time Computer Vision apps
 - It helps even better on mobile devices,
 - where we are usually power/performance bound
 - Memory transfers are very expensive, try to avoid it


