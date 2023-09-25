#!/usr/bin/env python3
import argparse
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
    "book",
    "book_docker",
    "serve_book",
]

# Build types.
BUILD_TYPES = ["Release", "RelWithDebInfo", "Debug", "Asan"]

# Some constants
SARA_SOURCE_DIR = pathlib.Path(__file__).parent.resolve()
SARA_DOCKER_IMAGE = "registry.gitlab.com/do-cv/sara"
SYSTEM = platform.system()

# Third-party libraries that makes Sara faster, stronger, cooler...
if SYSTEM == "Linux":
    HALIDE_ROOT_PATH = pathlib.Path.home() / "opt/Halide-16.0.0-x86-64-linux"
    ONNXRUNTIME_ROOT_PATH = (
        pathlib.Path.home() / "opt/onnxruntime-linux-x64-gpu-1.14.0"
    )
    NVIDIA_CODEC_SDK_ROOT_PATH = (
        pathlib.Path.home() / "opt/Video_Codec_SDK_12.1.14"
    )
    SWIFT_TOOLCHAIN_DIR = (
        pathlib.Path.home()
        / "opt/swift-5.9-RELEASE-ubuntu22.04"
    )
    SWIFT_TOOLCHAIN_BIN_DIR = SWIFT_TOOLCHAIN_DIR / "usr/bin"
    SWIFTC_PATH = SWIFT_TOOLCHAIN_BIN_DIR / "swiftc"
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


def infer_project_type(system: str):
    if system == "Linux":
        return "Ninja"
    elif system == "Darwin":
        return "Xcode"


PROJECT_TYPE = infer_project_type(SYSTEM)


def generate_project(
    source_dir: str,
    build_dir: str,
    build_type: str,
    from_scratch: bool = False,
):
    if from_scratch and build_dir.exists():
        shutil.rmtree(build_dir)

    if not build_dir.exists():
        pathlib.Path.mkdir(build_dir)

    cmake_options = []
    cmake_options.append("-G {}".format(PROJECT_TYPE))
    if PROJECT_TYPE != "Xcode":
        cmake_options.append("-D CMAKE_BUILD_TYPE={}".format(build_type))

    if SYSTEM == "Linux":
        cxx_compiler = SWIFT_TOOLCHAIN_BIN_DIR / "clang++"
        c_compiler = SWIFT_TOOLCHAIN_BIN_DIR / "clang"
        swift_bridging_include_dirs = SWIFT_TOOLCHAIN_DIR / "usr/include"
        cmake_options.append("-D CMAKE_CXX_COMPILER={}".format(cxx_compiler))
        cmake_options.append("-D CMAKE_C_COMPILER={}".format(c_compiler))
        cmake_options.append("-D CMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld")
        cmake_options.append("-D SWIFT_BRIDGING_INCLUDE_DIR={}".format(
            swift_bridging_include_dirs))

    # Support for YouCompleteMe auto-completions
    cmake_options.append("-D CMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON")

    # Use latest Qt version instead of the system Qt if possible.
    my_cmake_prefix_paths = []
    cmake_options.append("-D SARA_USE_QT6:BOOL=ON")
    if SYSTEM == "Darwin":
        qt6_root_dir = subprocess.check_output(["brew", "--prefix", "qt"])
        qt6_root_dir = qt6_root_dir.decode(sys.stdout.encoding).strip()
        qt6_cmake_dir = pathlib.Path(qt6_root_dir) / "lib" / "cmake" / "Qt6"
        cmake_options.append("-D Qt6_DIR={}".format(qt6_cmake_dir))

    # Compile shared or static libraries.
    cmake_options.append("-D SARA_BUILD_SHARED_LIBS:BOOL=ON")
    cmake_options.append("-D SARA_BUILD_TESTS:BOOL=ON")
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
        llvm_dir = subprocess.check_output(["brew", "--prefix", "llvm"])
        llvm_dir = llvm_dir.decode(sys.stdout.encoding).strip()
        llvm_cmake_dir = pathlib.Path(llvm_dir) / "lib" / "cmake" / "llvm"
        cmake_options.append("-D LLVM_DIR={}".format(llvm_cmake_dir))

    # Compile ONNX runtime code.
    if SYSTEM == "Linux" and ONNXRUNTIME_ROOT_PATH.exists():
        my_cmake_prefix_paths.append(ONNXRUNTIME_ROOT_PATH)

    # Compile nVidia platform's accelerated VideoIO.
    if (
        NVIDIA_CODEC_SDK_ROOT_PATH is not None
            and pathlib.Path(NVIDIA_CODEC_SDK_ROOT_PATH).exists()
    ):
        cmake_options.append(
            "-D NvidiaVideoCodec_ROOT={}".format(NVIDIA_CODEC_SDK_ROOT_PATH)
        )

    # Specify the paths for Qt and Halide.
    if my_cmake_prefix_paths:
        my_cmake_prefix_paths = [str(path) for path in my_cmake_prefix_paths]
        my_cmake_prefix_paths = ";".join(my_cmake_prefix_paths)
        cmake_options.append(
            "-D CMAKE_PREFIX_PATH={}".format(my_cmake_prefix_paths)
        )

    # Setup Swift bindings.
    if SYSTEM == "Darwin":
        cmake_options.append("-D CMAKE_Swift_COMPILER=/usr/bin/swiftc")
    elif SYSTEM == "Linux" and pathlib.Path(SWIFTC_PATH).exists():
        cmake_options.append("-D CMAKE_Swift_COMPILER={}".format(SWIFTC_PATH))

    # Setup Python bindings.
    if PYBIND11_DIR is not None:
        cmake_options.append("-D SARA_BUILD_PYTHON_BINDINGS:BOOL=ON")
        cmake_options.append("-D pybind11_DIR={}".format(PYBIND11_DIR))
        cmake_options.append(
            "-D PYTHON_INCLUDE_DIR={}".format(PYTHON_INCLUDE_DIR)
        )
        cmake_options.append("-D PYTHON_LIBRARY={}".format(PYTHON_LIBRARY))

    cmd = ["cmake", source_dir] + cmake_options
    execute(cmd, build_dir)


def build_project(build_dir: str, build_type: str):
    command_line = ["cmake", "--build", ".", "-j12", "-v"]
    if PROJECT_TYPE == "Xcode":
        command_line += ["--config", build_type]

    execute(command_line, build_dir)


def run_project_tests(build_dir: str, build_type: str):
    command_line = ["ctest", "--output-on-failure"]
    if PROJECT_TYPE == "Xcode":
        command_line += ["--config", build_type]

    execute(command_line, build_dir)


def build_library_docker(source_dir: str) -> None:
    # Build the docker image.
    execute(
        [
            "docker",
            "build",
            "-f",
            "docker/Dockerfile",
            "-t",
            f"{SARA_DOCKER_IMAGE}:latest",
            ".",
        ],
        source_dir,
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
            "{}:latest".format(sara_book_build_image),
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
            "{}:/workspace/book".format(SARA_SOURCE_DIR / "doc" / "book"),
            sara_book_build_image,
            "/bin/bash",
        ],
        cwd=(SARA_SOURCE_DIR / "doc" / "book"),
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
    args = parser.parse_args()

    for task in args.tasks:
        if task == "compilation_database":
            # Override the following options.
            PROJECT_TYPE = "Ninja"
            args.build_type = "Debug"

            build_dir = SARA_SOURCE_DIR.parent / "{}-build-{}".format(
                SARA_SOURCE_DIR.name, args.build_type
            )
            # Only regenerate the project
            generate_project(
                SARA_SOURCE_DIR, build_dir, args.build_type, args.from_scratch
            )
            # ... but don't build it.

        if task == "library":
            if PROJECT_TYPE == "Xcode":
                build_dir = SARA_SOURCE_DIR.parent / "{}-build-Xcode".format(
                    SARA_SOURCE_DIR.name
                )
            else:
                build_dir = SARA_SOURCE_DIR.parent / "{}-build-{}".format(
                    SARA_SOURCE_DIR.name, args.build_type
                )
            generate_project(
                SARA_SOURCE_DIR, build_dir, args.build_type, args.from_scratch
            )
            build_project(build_dir, args.build_type)
            run_project_tests(build_dir, args.build_type)

        if task == "library_docker":
            build_library_docker(SARA_SOURCE_DIR)

        if task == "book":
            build_book()

        if task == "book_docker":
            build_book_docker()

        if task == "serve_book":
            serve_book()
