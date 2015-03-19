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

1. C++ compiler
```
    $ sudo apt-get install g++
```
2. GNU make
```
    $ sudo apt-get install make
```

3. OpenCV for Tegra
```
    $ wget http://developer.download.nvidia.com/embedded/OpenCV/L4T_21.2/libopencv4tegra-repo_l4t-r21_2.4.10.1_armhf.deb
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