#!/usr/bin/env python3
import argparse
import pathlib
import platform
import shutil
import subprocess

BUILD_TYPE="Release"

RUN_FROM_DOCKER = True

SARA_SOURCE_DIR = pathlib.Path(__file__).parent.resolve()
SARA_BUILD_DIR = (SARA_SOURCE_DIR.parent /
                  "{}-build-{}".format(SARA_SOURCE_DIR.name, BUILD_TYPE))

SYSTEM = platform.system()

HALIDE_ROOT_PATH = pathlib.Path.home() / "opt/Halide-14.0.0-x86-64-linux"
NVIDIA_CODEC_SDK_ROOT_PATH = pathlib.Path.home() / "opt/Video_Codec_SDK_11.0.10"
SWIFTC_PATH= pathlib.Path.home() / "opt/swift-5.6.2-RELEASE-ubuntu20.04/usr/bin/swiftc"

try:
    import pybind11
    PYBIND11_DIR = pybind11.get_cmake_dir()

    import distutils.sysconfig as sysconfig
    PYTHON_INCLUDE_DIR=sysconfig.get_python_inc()
    PYTHON_LIBRARY=sysconfig.get_config_var('LIBDIR')
except:
    PYBIND11_DIR = None


def generate_project(source_dir: str, build_dir: str, from_scratch: bool = False):
    if from_scratch and build_dir.exists():
        shutil.rmtree(build_dir)

    if not build_dir.exists():
        pathlib.Path.mkdir(build_dir)

    cmake_options = []
    cmake_options.append('-D CMAKE_BUILD_TYPE={}'.format(BUILD_TYPE))
    cmake_options.append('-G Ninja')
    if SYSTEM == "Linux":
        cmake_options.append("-D CMAKE_EXE_LINKER_FLAGS=-fuse-ld=gold")

    # Support for YouCompleteMe auto-completions
    cmake_options.append("-D CMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON")

    # Use latest Qt version instead of the system Qt.
    cmake_options.append("-D SARA_USE_QT6:BOOL=ON")
    if SYSTEM == "Linux":
        my_cmake_prefix_paths = ["/usr/local/Qt-6.3.1"]
    elif SYSTEM == "Darwin":
        cmake_options.append("-D Qt6_DIR=$(brew --prefix qt)/lib/cmake/Qt6")

    # Compile shared or static libraries.
    cmake_options.append("-D SARA_BUILD_SHARED_LIBS:BOOL=ON")
    cmake_options.append("-D SARA_BUILD_TESTS:BOOL=ON")
    cmake_options.append("-D SARA_BUILD_SAMPLES:BOOL=ON")

    # Compile the Video I/O module.
    cmake_options.append("-D SARA_BUILD_VIDEOIO:BOOL=ON")

    # Compile Halide code.
    cmake_options.append("-D SARA_USE_HALIDE:BOOL=ON")
    if SYSTEM == "Linux" and HALIDE_ROOT_PATH.exists():
        my_cmake_prefix_paths.append(HALIDE_ROOT_PATH)
    elif SYSTEM == "Darwin":
        cmake_options.append("-D LLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm ")

    # Compile nVidia platform's accelerated VideoIO.
    if pathlib.Path(NVIDIA_CODEC_SDK_ROOT_PATH).exists():
        cmake_options.append("-D NvidiaVideoCodec_ROOT={}".format(NVIDIA_CODEC_SDK_ROOT_PATH))

    # Specify the paths for Qt and Halide.
    my_cmake_prefix_paths = [str(path) for path in my_cmake_prefix_paths]
    my_cmake_prefix_paths = ";".join(my_cmake_prefix_paths)
    cmake_options.append("-D CMAKE_PREFIX_PATH={}".format(my_cmake_prefix_paths))

    # Setup Swift bindings.
    if SYSTEM == "Darwin":
        cmake_options.append("-D CMAKE_Swift_COMPILER=$(which swiftc)")
    elif SYSTEM == "Linux" and pathlib.Path(SWIFTC_PATH).exists():
        cmake_options.append("-D CMAKE_Swift_COMPILER={}".format(SWIFTC_PATH))

    # Setup Python bindings.
    if PYBIND11_DIR is not None:
        cmake_options.append("-D SARA_BUILD_PYTHON_BINDINGS:BOOL=ON")
        cmake_options.append("-D pybind11_DIR={}".format(PYBIND11_DIR))
        cmake_options.append("-D PYTHON_INCLUDE_DIR={}".format(PYTHON_INCLUDE_DIR))
        cmake_options.append("-D PYTHON_LIBRARY={}".format(PYTHON_LIBRARY))

    subprocess.Popen(
        ['cmake', source_dir] + cmake_options,
        cwd=build_dir
    ).wait()


def build(build_dir: str):
    ret = subprocess.Popen(['cmake', '--build', '.', '-j12'],
                           cwd=build_dir).wait()
    return ret


generate_project(SARA_SOURCE_DIR, SARA_BUILD_DIR, True)
build(SARA_BUILD_DIR)
