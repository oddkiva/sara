#!/bin/bash
set -ex


if [ -z "$1" ]; then
  build_type=Release;
else
  build_type=$1;
fi

platform_name=$(uname -s)


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
  if [[ "${platform_name}" == "Darwin" ]] &&
     [[ "${build_type}" == "Xcode" ]]; then
    # Workaround for Xcode generator on Apple platforms.
    cmake_options+="-DCMAKE_C_COMPILER=$(which clang) "
    cmake_options+="-DCMAKE_CXX_COMPILER=$(which clang++) "

  elif [ "${platform_name}" == "Linux" ]; then
    local os_name=$(lsb_release -is)
    local os_version=$(lsb_release -rs)

    # I really want C++17.
    if [[ ${os_name} == "Ubuntu" ]] && [[ ${os_version} == "16.04" ]]; then
      cmake_options+="-DCMAKE_C_COMPILER=$(which gcc-7) "
      cmake_options+="-DCMAKE_CXX_COMPILER=$(which g++-7) "
    fi
  fi

  if [ "${platform_name}" == "Darwin" ]; then
    cmake_options+="-DQt5_DIR=$(brew --prefix qt)/lib/cmake/Qt5 "
  else
    cmake_options+="-DCMAKE_PREFIX_PATH=/home/david/Qt/5.12.6/gcc_64 "
  fi

  cmake_options+="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON "
  cmake_options+="-DSARA_BUILD_VIDEOIO=ON "
  cmake_options+="-DSARA_BUILD_PYTHON_BINDINGS=OFF "
  cmake_options+="-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") "
  cmake_options+="-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") "
  cmake_options+="-DSARA_BUILD_SHARED_LIBS=ON "
  cmake_options+="-DSARA_BUILD_TESTS=ON "
  cmake_options+="-DSARA_BUILD_SAMPLES=ON "

  cmake_options+="-DSARA_USE_HALIDE=ON "
  if [ "${platform_name}" == "Darwin" ]; then
    cmake_options+="-DHALIDE_DISTRIB_DIR=/usr/local "
  else
    cmake_options+="-DHALIDE_DISTRIB_DIR=/opt/halide "
  fi
  cmake_options+="-DNvidiaVideoCodec_ROOT=/opt/Video_Codec_SDK_9.1.23"

  # Generate makefile project.
  if [ "${build_type}" == "emscripten" ]; then
    emconfigure cmake ../sara
  else
    cmake ../sara ${cmake_options}
  fi

  # Build the library.
  cmake --build . -j$(nproc) -v

  # Run C++ tests.
  export BOOST_TEST_LOG_LEVEL=all
  export BOOST_TEST_COLOR_OUTPUT=1

  local test_options="--output-on-failure "
  if [[ "${build_type}" == "Xcode" ]]; then
    test_options+="-C Debug"
  fi
  ctest ${test_options}

  # Run Python tests.
  make pytest
  make package
}

function build_library_for_ios()
{
  # Generate an Xcode project.
  local cmake_options="-G Xcode "

  # Specific options for iOS.
  #
  # Build for ARM64 only.
  cmake_options+="-DCMAKE_SYSTEM_NAME=iOS "
  cmake_options+="-DCMAKE_OSX_ARCHITECTURES=arm64 "
  cmake_options+="-DCMAKE_OSX_DEPLOYMENT_TARGET=14.2 "
  cmake_options+="-DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=YES "
  cmake_options+="-DCMAKE_IOS_INSTALL_COMBINED=YES "

  # Workaround for Xcode generator on Apple platforms.
  cmake_options+="-DCMAKE_C_COMPILER=$(which clang) "
  cmake_options+="-DCMAKE_CXX_COMPILER=$(which clang++) "

  # For YouCompleteMe.
  cmake_options+="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON "

  # We need static builds for iOS.
  cmake_options+="-DSARA_BUILD_SHARED_LIBS=OFF "
  cmake_options+="-DBoost_INCLUDE_DIR=/usr/local/include "

  # Vanilla stuff.
  cmake_options+="-DSARA_BUILD_TESTS=ON "
  cmake_options+="-DSARA_BUILD_SAMPLES=ON "
  cmake_options+="-DSARA_USE_HALIDE=ON "
  if [ "${platform_name}" == "Darwin" ]; then
    cmake_options+="-DHALIDE_DISTRIB_DIR=/usr/local "
  else
    cmake_options+="-DHALIDE_DISTRIB_DIR=/opt/halide "
  fi

  # Generate the Xcode project.
  cmake ../sara ${cmake_options}

  # Build the library.
  cmake --build . -j$(nproc) -v

  # Run C++ tests.
  export BOOST_TEST_LOG_LEVEL=all
  export BOOST_TEST_COLOR_OUTPUT=1

  local test_options="--output-on-failure "
  if [[ "${build_type}" == "Xcode" ]]; then
    test_options+="-C Debug"
  fi
  ctest ${test_options}

  # Run Python tests.
  cmake --build . --target pytest
  cmake --build . --target  package
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
  if [[ ${build_type} == "ios" ]]; then
    build_library_for_ios
  else
    install_python_packages_via_pip
    build_library
  fi
  #install_package
}
cd ..
