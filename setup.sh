#!/bin/sh

sudo apt-get install g++ make cmake
wget http://developer.download.nvidia.com/embedded/OpenCV/L4T_21.2/libopencv4tegra-repo_l4t-r21_2.4.10.1_armhf.deb
sudo dpkg -i libopencv4tegra-repo_l4t-r21_2.4.10.1_armhf.deb
sudo apt-get install libopencv4tegra libopencv4tegra-dev
