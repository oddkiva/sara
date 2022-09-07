#!/bin/bash
set -ex


SARA_DOCKER_IMAGE=registry.gitlab.com/do-cv/sara


if [ -z "$1" ]; then
  build_type=Release;
else
  build_type=$1;
fi

platform_name=$(uname -s)

function install_python_packages_via_pip()
{
  pip3 install -r ../sara/requirements.txt
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
  cmake_options+="-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON "

  # We need static builds for iOS.
  cmake_options+="-DSARA_BUILD_SHARED_LIBS:BOOL=OFF "
  cmake_options+="-DBoost_INCLUDE_DIR=/usr/local/include "

  # Vanilla stuff.
  cmake_options+="-DSARA_BUILD_TESTS:BOOL=ON "
  cmake_options+="-DSARA_BUILD_SAMPLES:BOOL=ON "
  cmake_options+="-DSARA_USE_HALIDE:BOOL=ON "

  # Generate the Xcode project.
  time cmake ../sara ${cmake_options}

  # Build the library.
  time cmake --build . -j$(nproc) -v

  # Run C++ tests.
  export BOOST_TEST_LOG_LEVEL=all
  export BOOST_TEST_COLOR_OUTPUT=1

  local test_options="--output-on-failure "  # " -T memcheck"
  if [[ "${build_type}" == "Xcode" ]]; then
    test_options+="-C Debug"
  fi
  time ctest ${test_options}

  time cmake --build . --target package
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
    cmake --build . --target package --config Release
    cp libDO-Sara-shared-*-Darwin.tar.gz /Users/david/GitLab/DO-CV/sara-install
  fi
}


if [[ ${build_type} == "docker" ]]; then
  # Build the docker image.
  docker build -f docker/Dockerfile.bionic -t ${SARA_DOCKER_IMAGE}:latest .
  # Run the docker image.
  docker run --gpus all -it \
    -v $HOME/.gitconfig:/etc/gitconfig \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
    -v /media/Linux\ Data:/media/Linux\ Data \
    -v $PWD:/workspace/sara \
    -e DISPLAY \
    --ipc=host \
    --net=host \
    ${SARA_DOCKER_IMAGE} \
    /bin/zsh
else
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
    # install_package
  }
  cd ..
fi
