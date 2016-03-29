#!/bin/bash

set -e

if [[ $# == 0 ]]; then
  sara_build_dir="sara-build"
else
  sara_build_dir=$1
fi

# Create the build directory.
if [ -d "../${sara_build_dir}" ]; then
  rm -rf ../${sara_build_dir}
fi
mkdir ../${sara_build_dir}

cd ../${sara_build_dir}
{
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
    dpkg-sig --sign builder libDO-Sara-shared-*.deb
    sudo cp libDO-Sara-shared-*.deb /usr/local/debs
    sudo update-local-debs
    sudo apt-get update
    sudo apt-get install --reinstall libdo-sara-shared
  else
    rpm_package_name=$(echo `ls *.rpm`)
    sudo rpm -ivh --force ${rpm_package_name}
  fi
}
cd ..
