#!/bin/sh

#
# Copyright (c) 2014, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#

CPUFREQ=2320500
GPUFREQ=852000000
EMCFREQ=924000000


# Disable power management
#stop usdwatchdog
#stop ussrd
#setprop ussrd.enabled -1
#setprop ctl.stop ussrd
echo 0 > /sys/module/qos/parameters/enable
echo 0 > /sys/module/cpu_tegra/parameters/cpu_user_cap
#am force-stop com.nvidia.NvCPLSvc


# Fix the CPU frequency
echo 0 > /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable
echo 1 > /sys/kernel/cluster/immediate
echo 1 > /sys/kernel/cluster/force
echo G > /sys/kernel/cluster/active
#hotplug 1 1
#hotplug 2 1
#hotplug 3 1
echo 1 > /sys/devices/system/cpu/cpu1/online
echo 1 > /sys/devices/system/cpu/cpu2/online
echo 1 > /sys/devices/system/cpu/cpu3/online
echo userspace > /sys/devices/system/cpu/cpuquiet/current_governor
echo 0 > /sys/kernel/debug/edp/vdd_cpu/edp_reg_override
echo "userspace" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo "userspace" > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
echo "userspace" > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
echo "userspace" > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
echo $CPUFREQ > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
echo $CPUFREQ > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
echo $CPUFREQ > /sys/devices/system/cpu/cpu2/cpufreq/scaling_setspeed
echo $CPUFREQ > /sys/devices/system/cpu/cpu3/cpufreq/scaling_setspeed


# Fix the GPU frequency
echo $GPUFREQ > /sys/kernel/debug/clock/override.gbus/rate
echo 1 > /sys/kernel/debug/clock/override.gbus/state
echo $GPUFREQ > /sys/kernel/debug/clock/cap.gbus/rate


# Fix the memory controller clock
echo 1 > /sys/kernel/debug/clock/override.emc/state
echo $EMCFREQ > /sys/kernel/debug/clock/override.emc/rate


# Display the legal values for the GPU and memory clocks
cat /sys/kernel/debug/clock/gpu_dvfs_t  # displays legal GPU freqs (in Hz)
cat /sys/kernel/debug/clock/emc/possible_rates  # displays legal memory controller freqs (in kHz)
