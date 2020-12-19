#!/bin/bash
set -ex

sudo apt-get update -y -qq

# # Install the latest CMake please.
# sudo apt-get install -y -qq \
#   apt-transport-https \
#   ca-certificates \
#   gnupg \
#   software-properties-common \
#   wget
# wget -O \
#   - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
#   | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
# sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'

# All the packages to compile the C++ codebase.
apt_packages=(
  "build-essential"
  "ccache"
  "cmake"
  "cppcheck"
  "doxygen"
  "git"
  "graphviz"
  "lcov"
  "libboost-all-dev"
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
  "python3-dev"
  "qtbase5-dev"
)
apt_packages_str=$(printf "%s " "${apt_packages[@]}")
sudo apt-get install -y -qq ${apt_packages_str}

# Build CLBlast.
./scripts/install_clblast.sh
