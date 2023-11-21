#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
import pathlib
import platform
import shutil
import subprocess
import sys


# Build tasks
BUILD_TASKS = [
    "compilation_database",
    "library",
    "library_docker",
    "web",
    "book",
    "book_docker",
    "serve_book",
    "emsdk_docker",
]

# Build types.
BUILD_TYPES = ["Release", "RelWithDebInfo", "Debug", "Asan"]

# Platform and third-party version constants
UBUNTU_VERSION = "22.04"
CUDA_VERSION = "12.1.0"
TRT_VERSION = "8.6"
SWIFT_VERSION = "5.9"
HALIDE_VERSION = "16.0.0"

# Docker
SARA_SOURCE_DIR = pathlib.Path(__file__).parent.resolve()
SARA_DOCKER_IMAGE_BASENAME = "oddkiva/sara-devel"
SARA_DOCKER_IMAGE_VERSION = "-".join([f'cuda{CUDA_VERSION}',
                                      f'ubuntu{UBUNTU_VERSION}',
                                      f'trt{TRT_VERSION}',
                                      f'swift{SWIFT_VERSION}',
                                      f'halide{HALIDE_VERSION}'])
SARA_DOCKER_IMAGE = f'{SARA_DOCKER_IMAGE_BASENAME}:{SARA_DOCKER_IMAGE_VERSION}'

SYSTEM = platform.system()

FORCE_COMPILE_WITH_GCC = False

# Third-party libraries that makes Sara faster, stronger, cooler...
if SYSTEM == "Linux":
    OPT_PATH = pathlib.Path("/opt")
    HALIDE_ROOT_PATH = OPT_PATH / (f"Halide-{HALIDE_VERSION}-x86-64-linux")
    ONNXRUNTIME_ROOT_PATH = OPT_PATH / "onnxruntime-linux-x64-gpu-1.14.0"
    NVIDIA_CODEC_SDK_ROOT_PATH = OPT_PATH / "Video_Codec_SDK_12.1.14"
    if not FORCE_COMPILE_WITH_GCC:
        SWIFT_TOOLCHAIN_DIR = OPT_PATH / f"swift-{SWIFT_VERSION}-RELEASE-ubuntu{UBUNTU_VERSION}"
        SWIFT_TOOLCHAIN_BIN_DIR = SWIFT_TOOLCHAIN_DIR / "usr/bin"
        SWIFTC_PATH = SWIFT_TOOLCHAIN_BIN_DIR / "swiftc"
    else:
        SWIFTC_PATH = ""
elif SYSTEM == "Darwin":
    NVIDIA_CODEC_SDK_ROOT_PATH = None
    SWIFT_PATH = subprocess.check_output(["which", "swift"])

try:
    import pybind11

    PYBIND11_DIR = pybind11.get_cmake_dir()

    import distutils.sysconfig as sysconfig

    PYTHON_INCLUDE_DIR = sysconfig.get_python_inc()
    PYTHON_LIBRARY = sysconfig.get_config_var("LIBDIR")
except:
    PYBIND11_DIR = None


class BuildConfiguration:

    def __init__(self, args):
        self._os_name = "ubuntu"
        self._os_version = UBUNTU_VERSION
        self._cuda_version = CUDA_VERSION
        self._trt_version = TRT_VERSION
        self._halide_version = HALIDE_VERSION
        self._source_dir = SARA_SOURCE_DIR

        # Quick'n'dirty
        if FORCE_COMPILE_WITH_GCC:
            self._compiler = "gcc"
        else:
            self._compiler = "clang"
        self._build_type = args.build_type

        config_list = [self._build_type]
        if FORCE_COMPILE_WITH_GCC:
            config_list += [
                f"{self._compiler}",
                # f"{self._os_name}{self._os_version}",
                # f"cuda-{self._cuda_version}",
                # f"trt-{self._trt_version}",
                # f"halide-{self._halide_version}",
            ]

        stringified_config = "-".join(config_list)
        self._build_dir = f"{SARA_SOURCE_DIR.name}-build-{stringified_config}"

    @staticmethod
    def infer_project_type(system: str):
        if system == "Linux":
            return "Ninja"
        elif system == "Darwin":
            return "Xcode"


PROJECT_TYPE = BuildConfiguration.infer_project_type(SYSTEM)


def execute(cmd, cwd):
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
        cwd=cwd,
    ) as p:
        for line in p.stdout:
            print(line, end="")  # process line here

    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args)


def generate_project(
    source_dir: str,
    build_dir: str,
    build_type: str,
    from_scratch: bool = False,
    build_for_ci: bool = False
):
    if from_scratch and build_dir.exists():
        shutil.rmtree(build_dir)

    if not build_dir.exists():
        pathlib.Path.mkdir(build_dir)

    cmake_options = []
    cmake_options.append(f"-G {PROJECT_TYPE}")
    if PROJECT_TYPE != "Xcode":
        cmake_options.append(f"-D CMAKE_BUILD_TYPE={build_type}")

    if SYSTEM == "Linux" and not FORCE_COMPILE_WITH_GCC:
        cxx_compiler = SWIFT_TOOLCHAIN_BIN_DIR / "clang++"
        c_compiler = SWIFT_TOOLCHAIN_BIN_DIR / "clang"
        swift_bridging_include_dirs = SWIFT_TOOLCHAIN_DIR / "usr/include"
        cmake_options.append(f"-D CMAKE_CXX_COMPILER={cxx_compiler}")
        cmake_options.append(f"-D CMAKE_C_COMPILER={c_compiler}")
        # cmake_options.append("-D CMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld")
        cmake_options.append(
            f"-D SWIFT_BRIDGING_INCLUDE_DIR={swift_bridging_include_dirs}"
        )


    # Support for YouCompleteMe auto-completions
    cmake_options.append("-D CMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON")

    # Use latest Qt version instead of the system Qt if possible.
    my_cmake_prefix_paths = []
    cmake_options.append("-D SARA_USE_QT6:BOOL=ON")
    if SYSTEM == "Darwin":
        qt6_root_dir = subprocess.check_output(["brew", "--prefix", "qt"])
        qt6_root_dir = qt6_root_dir.decode(sys.stdout.encoding).strip()
        qt6_cmake_dir = pathlib.Path(qt6_root_dir) / "lib" / "cmake" / "Qt6"
        cmake_options.append(f"-D Qt6_DIR={qt6_cmake_dir}")

    # Compile shared or static libraries.
    cmake_options.append("-D SARA_BUILD_SHARED_LIBS:BOOL=ON")
    cmake_options.append("-D SARA_BUILD_TESTS:BOOL=ON")
    if not build_for_ci:
        cmake_options.append("-D SARA_BUILD_SAMPLES:BOOL=ON")
        cmake_options.append("-D SARA_BUILD_DRAFTS:BOOL=ON")

    # Compile the Video I/O module.
    cmake_options.append("-D SARA_BUILD_VIDEOIO:BOOL=ON")

    # Compile Halide code.
    if SYSTEM == "Linux" and HALIDE_ROOT_PATH.exists():
        cmake_options.append("-D SARA_USE_HALIDE:BOOL=ON")
        my_cmake_prefix_paths.append(HALIDE_ROOT_PATH)
    elif SYSTEM == "Darwin":
        cmake_options.append("-D SARA_USE_HALIDE:BOOL=ON")
        llvm_dir = subprocess.check_output(["brew", "--prefix", "llvm@16"])
        llvm_dir = llvm_dir.decode(sys.stdout.encoding).strip()
        llvm_cmake_dir = pathlib.Path(llvm_dir) / "lib" / "cmake" / "llvm"
        cmake_options.append(f"-D LLVM_DIR={llvm_cmake_dir}")

    # Compile ONNX runtime code.
    if SYSTEM == "Linux" and ONNXRUNTIME_ROOT_PATH.exists():
        my_cmake_prefix_paths.append(ONNXRUNTIME_ROOT_PATH)

    # Compile nVidia platform's accelerated VideoIO.
    if (
        NVIDIA_CODEC_SDK_ROOT_PATH is not None
            and pathlib.Path(NVIDIA_CODEC_SDK_ROOT_PATH).exists()
    ):
        cmake_options.append(
            f"-D NvidiaVideoCodec_ROOT={NVIDIA_CODEC_SDK_ROOT_PATH}"
        )

    # Specify the paths for Qt and Halide.
    if my_cmake_prefix_paths:
        my_cmake_prefix_paths = [str(path) for path in my_cmake_prefix_paths]
        my_cmake_prefix_paths = ";".join(my_cmake_prefix_paths)
        cmake_options.append(
            f"-D CMAKE_PREFIX_PATH={my_cmake_prefix_paths}"
        )

    # Setup Swift bindings.
    if SYSTEM == "Darwin":
        cmake_options.append("-D CMAKE_Swift_COMPILER=/usr/bin/swiftc")
    elif SYSTEM == "Linux" and pathlib.Path(SWIFTC_PATH).exists() and not FORCE_COMPILE_WITH_GCC:
        cmake_options.append(f"-D CMAKE_Swift_COMPILER={SWIFTC_PATH}")

    # Setup Python bindings.
    if PYBIND11_DIR is not None:
        cmake_options.append("-D SARA_BUILD_PYTHON_BINDINGS:BOOL=ON")
        cmake_options.append(f"-D pybind11_DIR={PYBIND11_DIR}")
        cmake_options.append(f"-D PYTHON_INCLUDE_DIR={PYTHON_INCLUDE_DIR}")
        cmake_options.append(f"-D PYTHON_LIBRARY={PYTHON_LIBRARY}")

    cmd = ["cmake", source_dir] + cmake_options
    execute(cmd, build_dir)


def generate_project_for_web(
    build_dir: str,
    build_type: str,
    from_scratch: bool = False
):
    if from_scratch and build_dir.exists():
        shutil.rmtree(build_dir)

    if not build_dir.exists():
        pathlib.Path.mkdir(build_dir)

    cmake_options = []
    cmake_options.append(f"-G Ninja")
    cmake_options.append(f"-D CMAKE_BUILD_TYPE={build_type}")
    cmake_options.append("-D CMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON")

    cmd = ["emcmake", "cmake", SARA_SOURCE_DIR] + cmake_options
    execute(cmd, build_dir)


def build_project(build_dir: str, build_type: str):
    cpu_count = mp.cpu_count()
    command_line = ["cmake", "--build", ".", f"-j{cpu_count}", "-v"]
    if PROJECT_TYPE == "Xcode":
        command_line += ["--config", build_type]

    execute(command_line, build_dir)


def run_project_tests(build_dir: str, build_type: str,
                      build_for_ci: bool = False):
    command_line = ["ctest", "--output-on-failure"]
    if build_for_ci:
        # N.B.: actually this is specific to GitHub Actions:
        # See: https://github.com/oddkiva/sara/actions/runs/6333376713/job/17201395646
        #
        # - build machines don't support CUDA, Vulkan at runtime
        # - it is not clear how to run GUI tests.
        #
        # We will rely on the testing on local machines instead and not rely
        # too much on GHA.
        tests_excluded = [
            "test_core_ipc_cond1",
            "test_graphics_*",
            "test_halide_*",
            "test_vulkan_*",
            "test_visualization_feature_draw",
            "shakti_test_*",
            "test_ransac_*"
        ]
        command_line.append("--exclude-regex")
        command_line.append("|".join(tests_excluded))

    if PROJECT_TYPE == "Xcode":
        command_line += ["--config", build_type]

    execute(command_line, build_dir)


def build_library_docker() -> None:
    # Build the docker image.
    execute(
        [
            "docker",
            "build",
            "-f",
            "docker/Dockerfile",
            "-t",
            f"{SARA_DOCKER_IMAGE}",
            ".",
        ],
        SARA_SOURCE_DIR,
    )


def build_book():
    ret = subprocess.Popen(
        ["Rscript", "build.R"], cwd=SARA_SOURCE_DIR / "doc" / "book"
    ).wait()
    return ret


def serve_book():
    ret = subprocess.Popen(
        ["Rscript", "-e", "bookdown::serve_book()"],
        cwd=SARA_SOURCE_DIR / "doc" / "book",
    ).wait()
    return ret


def build_book_docker():
    # Build the docker image.
    sara_book_build_image = "sara-book-build"
    ret = subprocess.Popen(
        [
            "docker",
            "build",
            "-f",
            "./docker/Dockerfile.book",
            "-t",
            f"{sara_book_build_image}:latest",
            ".",
        ],
        cwd=SARA_SOURCE_DIR,
    ).wait()

    # Run the docker image.
    book_dir_path = SARA_SOURCE_DIR / "doc" / "book"
    ret = subprocess.Popen(
        [
            "docker",
            "run",
            "-it",
            "-v",
            f"{book_dir_path}:/workspace/book",
            sara_book_build_image,
            "/bin/bash",
        ],
        cwd=(SARA_SOURCE_DIR / "doc" / "book"),
    ).wait()


def build_emsdk_docker():
    # Build the docker image.
    sara_emsdk_build_image = "oddkiva/sara-emsdk-devel"
    ret = subprocess.Popen(
        [
            "docker",
            "build",
            "-f",
            "./docker/Dockerfile.emsdk",
            "-t",
            f"{sara_emsdk_build_image}:latest",
            ".",
        ],
        cwd=SARA_SOURCE_DIR,
    ).wait()

    # Run the docker image.
    ret = subprocess.Popen(
        [
            "docker",
            "run",
            "-it",
            "-v",
            f"{SARA_SOURCE_DIR}:/workspace/sara",
            sara_emsdk_build_image,
            "/bin/bash",
        ],
        cwd=SARA_SOURCE_DIR,
    ).wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sara Build Program Options")

    parser.add_argument(
        "--tasks",
        choices=BUILD_TASKS,
        default="library",
        nargs="+",
        help="Specify the list of build tasks",
    )
    parser.add_argument(
        "--build_type",
        choices=BUILD_TYPES,
        default="Release",
        help="CMake build type",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Rebuild the project from scratch",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Build only essential parts of the project for Continous Integration",
    )
    args = parser.parse_args()

    for task in args.tasks:
        if task == "compilation_database":
            # Override the following options.
            PROJECT_TYPE = "Ninja"
            args.build_type = "Debug"

            build_dir = (
                SARA_SOURCE_DIR.parent /
                f"{SARA_SOURCE_DIR.name}-build-{args.build_type}"
            )
            # Only regenerate the project
            generate_project(
                SARA_SOURCE_DIR, build_dir, args.build_type, args.from_scratch
            )
            # ... but don't build it.

        if task == "library":
            if PROJECT_TYPE == "Xcode":
                build_dir = (
                    SARA_SOURCE_DIR.parent /
                    f"{SARA_SOURCE_DIR.name}-build-Xcode"
                )
            else:
                build_config = BuildConfiguration(args)
                build_dir = SARA_SOURCE_DIR.parent / build_config._build_dir
            generate_project(
                SARA_SOURCE_DIR, build_dir,
                args.build_type,
                args.from_scratch,
                args.ci
            )
            build_project(build_dir, args.build_type)
            run_project_tests(build_dir, args.build_type, args.ci)

        if task == "web":
            # Make sure that you have done the following:
            # $ cd $PATH_TO_EMSDK
            # $ git pull
            # $ ./emsdk install latest
            # $ ./emsdk activate latest
            # $ source ./emsdk_env.sh

            build_dir = (
                SARA_SOURCE_DIR.parent /
                f"{SARA_SOURCE_DIR.name}-build-Emscripten-{args.build_type}"
            )
            generate_project_for_web(
                build_dir,
                args.build_type,
                args.from_scratch
            )
            build_project(build_dir, args.build_type)

        if task == "emsdk_docker":
            build_emsdk_docker()

        if task == "library_docker":
            build_library_docker()

        if task == "book":
            build_book()

        if task == "book_docker":
            build_book_docker()

        if task == "serve_book":
            serve_book()
