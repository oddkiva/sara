cmake_minimum_required(VERSION 2.8.7)

project(DO-CV)
set(DO_VERSION 1.0.0)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 OLD)
endif (POLICY CMP0020)

# Configure DO++: put the versions.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindDO.cmake.in
               ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindDO.cmake @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/DO/Defines.hpp.in
               ${CMAKE_CURRENT_SOURCE_DIR}/src/DO/Defines.hpp @ONLY)

# Set default build type to Release mode.
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the build type, options are: Debug, Release" FORCE)

# Set options.
option(DO_USE_FROM_SOURCE        "Use DO++ libraries from source" ON)
option(DO_USE_STATIC_LIBS        "Use DO++ static libraries" OFF)
option(DO_BUILD_UNIT_TESTS       "Build unit tests for DO++ libraries" ON)
option(DO_BUILD_SAMPLE_PROGRAMS  "Build sample programs using DO++ libraries" ON)
option(DO_BUILD_SHARED_LIBS      "Build shared libraries for DO++ libraries" OFF)
option(DO_BUILD_IMAGE_IO         "Build third-party image I/O libraries" ON)

# Go!
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(DO REQUIRED)
do_dissect_version()
do_get_os_info()

# Group projects by category.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set installation settings.
#include(PCH)
include(DOInstall)

# Build third-party software (google test, FLANN,...)
add_subdirectory(third-party)

# Temporary
include_directories(third-party)

# Setup unit test projects
if (DO_BUILD_UNIT_TESTS)
  enable_testing()
  add_subdirectory(test)
endif ()

# Setup unit test projects
if (DO_BUILD_SAMPLE_PROGRAMS)
  add_subdirectory(examples)
endif ()

# Make documentation
add_subdirectory(doc)

# Make packages
include(DOMakePackages)
