#!/bin/bash
set -ex

# Update the packages list.
apt-get update -y -qq

# Install the preliminary packages.
apt-get install -y -qq \
  apt-transport-https \
  ca-certificates \
  gnupg \
  software-properties-common \
  wget

# CMake APT repository.
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
  gpg --dearmor - | \
  tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
# Vulkan SDK APT repository.
wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -
wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list http://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list
# CLBlast APT repository.
add-apt-repository ppa:cnugteren/clblast

# Update the packages list again.
apt-get update -y -qq

# All the packages to compile the C++ codebase.
apt_packages=(
  # C++ toolchain and tools.
  "build-essential"
  "ccache"
  "cmake"
  "cppcheck"
  "git"
  "lcov"
  "ninja-build"
  "python3-pip"
  "rubygems"
  # Documentation
  "doxygen"
  "graphviz"
  # Libraries.
  "gstreamer1.0-plugins-base"
  "gstreamer1.0-plugins-good"
  "gstreamer1.0-plugins-bad"
  "gstreamer1.0-plugins-ugly"
  "gstreamer1.0-libav"
  "gstreamer1.0-doc"
  "gstreamer1.0-tools"
  "gstreamer1.0-x"
  "gstreamer1.0-alsa"
  "gstreamer1.0-gl"
  "gstreamer1.0-gtk3"
  "gstreamer1.0-qt5"
  "gstreamer1.0-pulseaudio"
  "libboost-all-dev"
  "libclblast-dev"
  "libgstreamer1.0-0"
  "libhdf5-dev"
  "libjpeg-dev"
  "libpng-dev"
  "libtiff5-dev"
  "libavcodec-dev"
  "libavformat-dev"
  "libavutil-dev"
  "libswscale-dev"
  "libglew-dev"
  "libglfw3-dev"
  "libceres-dev"
  "libpocl-dev"
  "libz3-4"  # For swiftc
  "ocl-icd-opencl-dev"
  "opencl-headers"
  "python3-dev"
  "qtbase5-dev"
  "vulkan-sdk"
)
apt_packages_str=$(printf "%s " "${apt_packages[@]}")

# Install all packages in a non-interactive way.
apt-get install -y ${apt_packages_str}
