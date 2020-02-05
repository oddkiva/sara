cmake_minimum_required(VERSION 3.10)

project(Sara LANGUAGES CXX)


# Set the version major and minor manually.
set(DO_Sara_VERSION_MAJOR 1)
set(DO_Sara_VERSION_MINOR 2)

if (APPLE)
  set (CMAKE_MACOSX_RPATH ON)
endif ()

# Check that vcpkg toolchain is specified when running CMake!
if (WIN32 AND NOT CMAKE_TOOLCHAIN_FILE)
  message(FATAL_ERROR
    "[Sara] Could not find vpkg toolchain file: "
    "please specify it to build Sara on Windows!")
endif ()

# Set default build type to Release mode.
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
  "Build type options are: Debug, Release")

# Set options.
option(SARA_USE_VLD "Enable Visual Leak Detector for unit tests" OFF)
option(SARA_USE_HALIDE "Enable Halide" OFF)
option(SARA_BUILD_VIDEOIO "Build Sara's Video I/O module" OFF)
option(SARA_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(SARA_BUILD_TESTS "Build unit tests for DO-Sara libraries" OFF)
option(SARA_BUILD_SAMPLES "Build sample programs using DO-Sara libraries" OFF)
option(SARA_BUILD_SHARED_LIBS "Build shared libraries for DO-Sara libraries" OFF)
option(SARA_SELF_CONTAINED_INSTALLATION
  "Install C++ and Python libraries in a single self contained directory" OFF)


# Set the following internal switches to build Sara.
set(SARA_USE_FROM_SOURCE TRUE CACHE INTERNAL "Build Sara libraries")
set(BUILD_SHARED_LIBS ${SARA_BUILD_SHARED_LIBS})

# Add Sara to the CMake module path.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


# We are in 2020 now: let's all look to the future please.
set(CMAKE_CXX_STANDARD 14)

# Use CUDA if available.
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else ()
  message(STATUS "No CUDA support!")
endif()
# set(CMAKE_CUDA_FLAGS -arch=sm_60)
# set(CMAKE_CUDA_FLAGS -arch=sm_75)
find_package(CUDA 10.0)
if (CUDA_FOUND)
  set(CUDA_VERBOSE_BUILD ON)
  include(shakti_macros)
endif ()

# Activate the usage of CCache to speed up builds if available.
# Configure Sara library version.
include(sara_macros)
sara_dissect_version()

include(sara_use_ccache)

# We are ready to build Sara.
find_package(DO_Sara REQUIRED)

# Group projects by category.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set installation settings.
include(sara_installation_settings)

# Enable testing before building the library.
if (SARA_BUILD_TESTS)
  include(sara_configure_code_coverage)
  include(CTest)
  enable_testing()
endif ()

# Build the libraries.
add_subdirectory(cpp)
if (SARA_BUILD_PYTHON_BINDINGS)
  add_subdirectory(python)
endif ()

# Make documentation
add_subdirectory(doc)

# To create Windows installer and Linux packages.
include(CPack)
