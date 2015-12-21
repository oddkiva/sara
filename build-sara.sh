#! /bin/bash

mkdir ../sara-build-shared
cd ../sara-build-shared
cmake ../sara \
  -DCMAKE_BUILD_TYPE=Release \
  -DSARA_BUILD_SHARED_LIBS=ON \
  -DSARA_BUILD_TESTS=ON \
  -DSARA_BUILD_SAMPLES=ON

make -j`nproc` && make test && make package

dpkg-sig --sign builder ../sara-build-shared/libDO-Sara-shared-*.deb
sudo cp ../sara-build-shared/libDO-Sara-shared-*.deb /usr/local/debs
sudo update-local-debs
sudo apt-get update
sudo apt-get install --reinstall libdo-sara-shared
