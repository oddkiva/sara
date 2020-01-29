#!/bin/bash
set -ex


if [ -z "$1" ]; then
  build_type=Release;
else
  build_type=$1;
fi

platform_name=$(uname -s)
os_name=$(lsb_release -is)
os_version=$(lsb_release -rs)


function install_python_packages_via_pip()
{
  pip install -r ../sara/requirements.txt
}

function build_library()
{
  if [ "${build_type}" == "Xcode" ]; then
    local cmake_options="-G Xcode "
  else
    local cmake_options="-DCMAKE_BUILD_TYPE=${build_type} "
  fi

  if [[ ${os_name} == "Ubuntu" ]] && [[ ${os_version} == "16.04" ]]; then
    cmake_options+="-DCMAKE_C_COMPILER=$(which gcc-7) "
    cmake_options+="-DCMAKE_CXX_COMPILER=$(which g++-7) "
  fi

  if [ "${platform_name}" == "Darwin" ]; then
    cmake_options+="-DQt5_DIR=$(brew --prefix qt)/lib/cmake/Qt5 "
  else
    cmake_options+="-DCMAKE_PREFIX_PATH=/home/david/Qt/5.12.6/gcc_64 "
  fi

  cmake_options+="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON "
  cmake_options+="-DSARA_BUILD_VIDEOIO=ON "
  cmake_options+="-DSARA_BUILD_PYTHON_BINDINGS=ON "
  cmake_options+="-DSARA_BUILD_SHARED_LIBS=ON "
  cmake_options+="-DSARA_BUILD_TESTS=ON "
  cmake_options+="-DSARA_BUILD_SAMPLES=ON "

  cmake_options+="-DSARA_USE_HALIDE=ON "
  cmake_options+="-DHALIDE_DISTRIB_DIR=/opt/halide"

  # Generate makefile project.
  if [ "${build_type}" == "emscripten" ]; then
    emconfigure cmake ../sara
  else
    cmake ../sara ${cmake_options}
  fi

  # Build the library.
  make -j$(nproc) VERBOSE=1

  # Run C++ tests.
  export BOOST_TEST_LOG_LEVEL=all
  export BOOST_TEST_COLOR_OUTPUT=1
  ctest --output-on-failure

  # Run Python tests.
  make pytest
  make package
}

function install_package()
{
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


sara_build_dir="sara-build-${build_type}"

# Create the build directory.
if [ -d "../${sara_build_dir}" ]; then
  rm -rf ../${sara_build_dir}
fi

mkdir ../${sara_build_dir}


cd ../${sara_build_dir}
{
  install_python_packages_via_pip
  build_library
  #install_package
}
cd ..
