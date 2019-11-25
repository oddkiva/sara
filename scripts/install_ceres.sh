#!/bin/bash
set -ex

# Build Ceres only once.
pushd ../cpp/third-party/ceres-solver-1.14.0
{
  mkdir build
  cd build
  cmake .. -DCXX11=ON
  make -j$(nproc)
  make install
}
popd
