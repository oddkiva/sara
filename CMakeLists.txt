cmake_minimum_required(VERSION 2.8.12)

project(Sara)

# Set the version major and minor manually.
set(DO_Sara_VERSION_MAJOR 1)
set(DO_Sara_VERSION_MINOR 2)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 OLD)
endif (POLICY CMP0020)

# Set default build type to Release mode.
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Build type options are: Debug, Release")

# Set options.
option(SARA_USE_VLD "Enable Visual Leak Detector for unit tests" OFF)
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

# Configure Sara library version.
include(sara_macros)
sara_dissect_version()

# We are ready to build Sara.
find_package(DO_Sara REQUIRED)


# Group projects by category.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set installation settings.
include(sara_installation_settings)

# Build third-party software (google test, FLANN,...)
add_subdirectory(third-party)

if (SARA_BUILD_PYTHON_BINDINGS)
  add_subdirectory(python)
endif ()

# Setup unit test projects
if (SARA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif ()

# Setup unit test projects
if (SARA_BUILD_SAMPLES)
  add_subdirectory(examples)
endif ()

# Make documentation
add_subdirectory(doc)

# To create Windows installer and Linux packages.
include(CPack)
