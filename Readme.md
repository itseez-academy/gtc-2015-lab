# Intro

`GTC2015_stitching` project provides simple panorama stitching pipeline that
allows various CPU and GPU oriented optimizations for Jetson TK1 platform.
Bottlenecks are:

- Feature matching step `cv::detail::BestOf2NearestMatcher::operator()`
- Seam search step `cv::detail::VoronoiSeamFinder::find`
- Blending step `cv::detail::SphericalWarper::warp`
  and `detail::MultiBandBlender::blend`

OpenCV provides GPU-based alternative for Feature matching and Blending steps that
give speedup from 1.5x to 2.5x to each step. `VoronoiSeamFinder::find` method
does not have GPU alternative, but switch to GPU pipeline allows significantly
improve performance of seam search procedure globally.

# Prerequisites

0. Get Demo source code and input images for stitching from GitHub
```
    $ git clone https://github.com/opencv/stitching-gtc2015
```
1. Setup development environment on your board

```
    $ cd stitching-gtc2015
    $ ./setup.sh
```
2. Turn of Tegra power management system to get reliable performance results.
```
    $ sudo ./freqScale_l4t.sh
```

# Build & Run from command line
```
    $ make
    $ ./stitching ./images/*.jpg
```

# Practice

By default application works on the CPU. And it has three major bottlenecks
mentioned above. Lab already contains timings that allows to determing
bottlenecks and all optimization steps. We will try to switch from CPU
version to GPU version with CUDA optimizations and optimize it for
Tegra TK1 platform.