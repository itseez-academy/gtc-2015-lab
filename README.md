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

SSH to the Jetson TK1 board:

    $ __TBD__

Get the source code and input images from GitHub:

    $ wget https://github.com/Itseez/gtc-2015-lab/archive/master.zip
    $ unzip master.zip

Setup development environment:

    $ cd ./gtc-2015-lab-master
    $ ./setup.sh

Turn off power management on Tegra in order to get reliable
performance measurements:

    $ sudo ./freqScale_l4t.sh

# Build & Run

Execute the following commands on your Jetson TK1 board:

    $ make
    $ ./stitching ./images/*.jpg

You will see the performance report in your console. You can also
check the final panorama exported to the `result.jpg` file.

# Working on the lab

By default the application works on CPU. And it has three major
bottleneck functions:

  - Feature matching `cv::detail::BestOf2NearestMatcher::operator()`
  - Seam finding `cv::detail::VoronoiSeamFinder::find`
  - Blending `cv::detail::SphericalWarper::warp` and `detail::MultiBandBlender::blend`

Source code contains time measurements that allow to detect
bottlenecks and all potential optimization steps. We will try to
switch from CPU to GPU version using CUDA and then tune it for the
Jetson TK1 platform.

OpenCV provides GPU-based implementations for Feature matching and
Blending steps that give speedup ranging from 1.5x to 2.5x.
`VoronoiSeamFinder::find` method does not have GPU implementation, but
switch to the GPU pipeline allows to significantly improve performance
of the seam search procedure in general.

## Detailed instructions

### Step 1. Profile CPU implementation

Run the app and record perf data

### Step 2. Switch to GPU matching

__TBD__

### Step 3. Switch to GPU pipeline

__TBD__
