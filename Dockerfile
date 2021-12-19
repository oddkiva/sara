FROM nvidia/cudagl:11.4.2-devel

MAINTAINER "Odd Kiva"

# To avoid console interaction with apt.
ARG DEBIAN_FRONTEND=noninteractive

# Install necessary packages to add APT repositories.
RUN apt-get update -y && apt-get install -y \
  apt-transport-https        \
  ca-certificates            \
  gnupg                      \
  software-properties-common \
  wget

# CMake APT repository.
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
  gpg --dearmor - | \
  tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'

# Vulkan SDK APT repository.
RUN wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -
RUN wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list http://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list

# CLBlast APT repository.
RUN add-apt-repository ppa:cnugteren/clblast

# C++ toolchain, libraries and tools.
RUN apt-get update -y && apt-get install -y \
  build-essential           \
  ccache                    \
  cmake                     \
  cppcheck                  \
  git                       \
  lcov                      \
  ninja-build               \
  python3-pip               \
  rubygems                  \
  doxygen                   \
  graphviz                  \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad  \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav        \
  gstreamer1.0-doc          \
  gstreamer1.0-tools        \
  gstreamer1.0-x            \
  gstreamer1.0-alsa         \
  gstreamer1.0-gl           \
  gstreamer1.0-gtk3         \
  gstreamer1.0-qt5          \
  gstreamer1.0-pulseaudio   \
  libboost-all-dev          \
  libclblast-dev            \
  libgstreamer1.0-0         \
  libhdf5-dev               \
  libjpeg-dev               \
  libpng-dev                \
  libtiff5-dev              \
  libavcodec-dev            \
  libavformat-dev           \
  libavutil-dev             \
  libswscale-dev            \
  libglew-dev               \
  libglfw3-dev              \
  libceres-dev              \
  libpocl-dev               \
  libz3-4                   \
  ocl-icd-opencl-dev        \
  opencl-headers            \
  python3-dev               \
  qtbase5-dev               \
  vulkan-sdk

# Install Halide.
RUN mkdir ${HOME}/opt
RUN wget https://github.com/halide/Halide/releases/download/v13.0.0/Halide-13.0.0-x86-64-linux-c3641b6850d156aff6bb01a9c01ef475bd069a31.tar.gz
RUN tar xvzf Halide-13.0.0-x86-64-linux-c3641b6850d156aff6bb01a9c01ef475bd069a31.tar.gz
RUN mv Halide-13.0.0-x86-64-linux ${HOME}/opt

# Install Swift toolchain.
RUN wget https://download.swift.org/swift-5.5.1-release/ubuntu2004/swift-5.5.1-RELEASE/swift-5.5.1-RELEASE-ubuntu20.04.tar.gz \
      && tar xvzf swift-5.5.1-RELEASE-ubuntu20.04.tar.gz \
      && mv swift-5.5.1-RELEASE-ubuntu20.04 ${HOME}/opt


# Install Python dependencies.
RUN pip3 install \
      coverage==4.5.4 \
      ipdb            \
      ipdbplugin      \
      nose            \
      numpy           \
      scipy           \
      PySide2         \
      ipython         \
      pybind11

# Please make my life easier
# TODO: install neovim, etc.
RUN apt-get install -y zsh

# Set up my development workspace.
WORKDIR /workspace/sara
