#!/bin/bash
set -ex


git clone https://github.com/CNugteren/CLBlast
git checkout 1.5.0
cd CLBlast
mkdir build
cd build
cmake ..
make -j$(nproc)
make install
