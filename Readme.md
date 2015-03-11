# Prerequisites

1. C++ compiler
```
    $ sudo apt-get install g++
```
2. GNU make
```
    $ sudo apt-get install make
```
3. CMake 2.8.7+
```
    $ sudo apt-get install cmake
```
4. OpenCV for Tegra
```
    $ wget http://developer.download.nvidia.com/embedded/OpenCV/L4T_21.2/libopencv4tegra-repo_l4t-r21_2.4.10.1_armhf.deb
    $ sudo dpkg -i libopencv4tegra-repo_l4t-r21_2.4.10.1_armhf.deb
    $ sudo apt-get install libopencv4tegra libopencv4tegra-dev
```

# Build

```
    $ cmake
    $ make
```
