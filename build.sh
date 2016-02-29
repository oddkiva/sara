#! /bin/bash
set -e

# Create the build directory.
if [ -d "../sara-build" ]; then
  rm -rf ../sara-build
fi
mkdir ../sara-build
cd ../sara-build

# Generate makefile project.
cmake ../sara \
  -DCMAKE_BUILD_TYPE=Release \
  -DSARA_BUILD_VIDEOIO=ON \
  -DSARA_BUILD_PYTHON_BINDINGS=ON \
  -DSARA_BUILD_SHARED_LIBS=ON \
  -DSARA_BUILD_TESTS=ON \
  -DSARA_BUILD_SAMPLES=ON

# Build the library.
make -j`nproc` && make test && make pytest && make package

if [ -f "/etc/debian_version" ]; then
  # Register the package to the local debian repository.
  dpkg-sig --sign builder ../sara-build-shared/libDO-Sara-shared-*.deb
  sudo cp ../sara-build-shared/libDO-Sara-shared-*.deb /usr/local/debs
  sudo update-local-debs
  sudo apt-get update
  sudo apt-get install --reinstall libdo-sara-shared
else
  rpm_package_name=$(echo `ls *.rpm`)
  sudo rpm -ivh --force ${rpm_package_name}
fi
