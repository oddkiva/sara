cmake_minimum_required(VERSION 2.8.12)

project(Sara CXX)

# Set the version major and minor manually.
set(DO_Sara_VERSION_MAJOR 1)
set(DO_Sara_VERSION_MINOR 2)

if (APPLE)
  set (CMAKE_MACOSX_RPATH ON)
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

# Activate the usage of CCache to speed up builds if available.
include(sara_use_ccache)

# Configure Sara library version.
include(sara_macros)
sara_dissect_version()

# We are ready to build Sara.
find_package(DO_Sara REQUIRED)


# Group projects by category.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set installation settings.
include(sara_installation_settings)

# Enable testing before building the library.
if (SARA_BUILD_TESTS)
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
