FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

LABEL authors="Odd Kiva"

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
      tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy-rc main' | \
      tee -a /etc/apt/sources.list.d/kitware.list >/dev/null

# Vulkan SDK APT repository.
RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | tee /etc/apt/trusted.gpg.d/lunarg.asc
RUN wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list http://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list

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
  libboost-all-dev          \
  libclblast-dev            \
  libhdf5-dev               \
  libheif-dev               \
  libjpeg-dev               \
  libpng-dev                \
  libtiff5-dev              \
  libwebp-dev               \
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
  libtbb-dev                \
  ocl-icd-opencl-dev        \
  opencl-headers            \
  python3-dev               \
  qt6-base-dev              \
  vulkan-sdk

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

# Install Swift toolchain.
RUN wget https://download.swift.org/swiftly/linux/swiftly-$(uname -m).tar.gz && \
      tar zxf swiftly-$(uname -m).tar.gz && \
      ./swiftly init --quiet-shell-followup && \
      . "${SWIFTLY_HOME_DIR:-$HOME/.local/share/swiftly}/env.sh" && \
      hash -r

# Install Halide.
RUN wget https://github.com/halide/Halide/releases/download/v19.0.0/Halide-19.0.0-x86-64-linux-5f17d6f8a35e7d374ef2e7e6b2d90061c0530333.tar.gz
RUN tar xvzf Halide-19.0.0-x86-64-linux-5f17d6f8a35e7d374ef2e7e6b2d90061c0530333.tar.gz && \
      mv Halide-19.0.0-x86-64-linux /opt

# Please make my life easier
# TODO: install neovim, etc.
RUN apt-get install -y zsh

# Set up my development workspace.
WORKDIR /workspace/sara
