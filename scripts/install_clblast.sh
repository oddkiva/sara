#!/bin/bash
set -ex


git clone https://github.com/CNugteren/CLBlast
cd CLBlast
git checkout 1.5.0
mkdir build
cd build
cmake ..
make -j$(nproc)
make install
