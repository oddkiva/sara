FROM nvidia/cudagl:11.4.2-runtime-ubuntu20.04

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
  cuda-toolkit-11-7         \
  git                       \
  lcov                      \
  ninja-build               \
  python3-pip               \
  rubygems                  \
  doxygen                   \
  graphviz                  \
  libboost-all-dev          \
  libclblast-dev            \
  libhdf5-dev               \
  libheif-dev               \
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
  libcurl4                  \
  libz3-4                   \
  ocl-icd-opencl-dev        \
  opencl-headers            \
  python3-dev               \
  qtbase5-dev               \
  vulkan-sdk

# Install Halide.
RUN mkdir ${HOME}/opt
RUN wget https://github.com/halide/Halide/releases/download/v14.0.0/Halide-14.0.0-x86-64-linux-6b9ed2afd1d6d0badf04986602c943e287d44e46.tar.gz
RUN tar xvzf Halide-14.0.0-x86-64-linux-6b9ed2afd1d6d0badf04986602c943e287d44e46.tar.gz
RUN mv Halide-14.0.0-x86-64-linux ${HOME}/opt

# Install Swift toolchain.
RUN wget https://download.swift.org/swift-5.6.3-release/ubuntu2004/swift-5.6.3-RELEASE/swift-5.6.3-RELEASE-ubuntu20.04.tar.gz \
      && tar xvzf swift-5.6.3-RELEASE-ubuntu20.04.tar.gz \
      && mv swift-5.6.3-RELEASE-ubuntu20.04 ${HOME}/opt


# Install Python dependencies.
RUN pip3 install \
      coverage    \
      ipdb        \
      ipdbplugin  \
      nose        \
      numpy       \
      scipy       \
      PySide2     \
      ipython     \
      pybind11

# Please make my life easier
# TODO: install neovim, etc.
RUN apt-get install -y zsh

# Set up my development workspace.
WORKDIR /workspace/sara
