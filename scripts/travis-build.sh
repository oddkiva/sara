#!/bin/bash
set -ex


if [ -z "$1" ]; then
  build_type=Release;
else
  build_type=$1;
fi


function install_python_packages_via_pip()
{
  pip install numpy nose
}

function build_library()
{
  local cmake_options="-DCMAKE_BUILD_TYPE=${build_type} "
  cmake_options+="-DSARA_BUILD_VIDEOIO=ON "
  #cmake_options+="-DSARA_BUILD_PYTHON_BINDINGS=ON "
  cmake_options+="-DSARA_BUILD_SHARED_LIBS=ON "
  cmake_options+="-DSARA_BUILD_TESTS=ON "
  cmake_options+="-DSARA_BUILD_SAMPLES=ON "
  cmake_options+="-DCMAKE_PREFIX_PATH=/root/ffmpeg "

  # Generate makefile project.
  cmake ../sara ${cmake_options}

  # Build the library.
  make -j$(nproc) VERBOSE=1

  # Run C++ tests.
  export BOOST_TEST_LOG_LEVEL=all
  export BOOST_TEST_COLOR_OUTPUT=1
  ctest --output-on-failure --exclude-regex \
    "test_graphics_*|test_features_draw"

  # Run Python tests.
  make pytest
  make package
}

function install_package()
{
  # Register the package to the local debian repository.
  dpkg-sig --sign builder libDO-Sara-shared-*.deb
  sudo cp libDO-Sara-shared-*.deb /usr/local/debs
  sudo update-local-debs
  sudo apt-get update
  sudo apt-get install --reinstall libdo-sara-shared
}


sara_build_dir="sara-build-${build_type}"

# Create the build directory.
if [ -d "${sara_build_dir}" ]; then
  rm -rf ${sara_build_dir}
fi
mkdir ${sara_build_dir}

cd ${sara_build_dir}
{
  build_library
}
cd ..
