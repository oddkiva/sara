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

function build_library()
{
  # ========================================================================= #
  # Specify the build type except for Xcode.
  #
  if [ "${build_type}" == "Xcode" ]; then
    local cmake_options="-G Xcode "
  else
    local cmake_options="-G Ninja "
    local cmake_options+="-DCMAKE_BUILD_TYPE=${build_type} "
  fi

  # Setup the C and C++ toolchain.
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

  # ========================================================================= #
  # Use the gold linker if available.
  if [ "$(uname -s)" == "Linux" ]; then
    cmake_options+="-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=gold "
  fi

  # ========================================================================= #
  # Support for YouCompleteMe code auto-completion.
  cmake_options+="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON "

  # ========================================================================= #
  # Setup Swift toolchain.
  if [[ "${platform_name}" == "Darwin" ]]; then
    cmake_options+="-DCMAKE_Swift_COMPILER=$(which swiftc) "
  elif [[ "${platform_name}" == "Linux" ]]; then
    SWIFTC_PATH="${HOME}/opt/swift-5.5.1-RELEASE-ubuntu20.04/usr/bin/swiftc"
    if [ -f "${SWIFTC_PATH}" ]; then
      cmake_options+="-DCMAKE_Swift_COMPILER=${SWIFTC_PATH} "
    fi

    PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
    cmake_options+="-Dpybind11_DIR=${PYBIND11_DIR} "
  fi

  # Use latest Qt version instead of the system Qt.
  #
  # TODO: migrate to Qt6.
  # if [ "${platform_name}" == "Linux" ]; then
  #   cmake_options+="-DQt5_DIR=${HOME}/opt/Qt-5.15.0-amd64/lib/cmake/Qt5 "
  if [ "${platform_name}" == "Darwin" ]; then
    cmake_options+="-DSARA_USE_QT6=ON "
    cmake_options+="-DQt6_DIR=$(brew --prefix qt)/lib/cmake/Qt6 "
  fi

  # ========================================================================= #
  # Sara specific options.
  #
  # Compile the Video I/O module.
  cmake_options+="-DSARA_BUILD_VIDEOIO=ON "
  cmake_options+="-DSARA_BUILD_PYTHON_BINDINGS=ON "
  cmake_options+="-DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") "
  cmake_options+="-DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") "

  # Compile shared or static libraries.
  cmake_options+="-DSARA_BUILD_SHARED_LIBS=ON "
  cmake_options+="-DSARA_BUILD_TESTS=ON "
  cmake_options+="-DSARA_BUILD_SAMPLES=ON "

  # Compile Halide code.
  cmake_options+="-DSARA_USE_HALIDE=ON "
  if [ "${platform_name}" == "Linux" ]; then
    cmake_options+="-DCMAKE_PREFIX_PATH=$HOME/opt/Halide-13.0.0-x86-64-linux "
  fi
  if [ "${platform_name}" == "Darwin" ]; then
    cmake_options+="-DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm "
  fi

  # nVidia platform's specific options.
  if [ -d "${HOME}/opt/Video_Codec_SDK_11.0.10" ]; then
    cmake_options+="-DNvidiaVideoCodec_ROOT=${HOME}/opt/Video_Codec_SDK_11.0.10 "
  fi

  echo $(which cmake)
  echo $(cmake --version)


  # ========================================================================= #
  # Now generate the makefile project.
  if [ "${build_type}" == "emscripten" ]; then
    emconfigure cmake ../sara
  else
    time cmake ../sara ${cmake_options} \
      --profiling-format=google-trace \
      --profiling-output=$(pwd)/cmake-sara.log
  fi

  # Build the library.
  time cmake --build . -j$(nproc)

  # Run C++ tests.
  export BOOST_TEST_LOG_LEVEL=all
  export BOOST_TEST_COLOR_OUTPUT=1

  local test_options="--output-on-failure "
  if [[ "${build_type}" == "Xcode" ]]; then
    test_options+="-C Debug"
  fi
  time ctest ${test_options}

  # Run Python tests.
  time cmake --build . --target pytest
  time cmake --build . --target package
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

  # Run Python tests.
  time cmake --build . --target pytest
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
  docker build -f Dockerfile -t ${SARA_DOCKER_IMAGE}:latest .
  # Run the docker image.
  docker run --gpus all -it \
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
