cmake_minimum_required(VERSION 2.8.7)

set(DO_PROJECT_NAME Sara CACHE STRING "DO Project")
project(${DO_PROJECT_NAME})
set(DO_${DO_PROJECT_NAME}_VERSION 1.1.0)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 OLD)
endif (POLICY CMP0020)

# Configure DO++: put the versions.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DO_${DO_PROJECT_NAME}_version.cmake.in
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DO_${DO_PROJECT_NAME}_version.cmake @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DO/${DO_PROJECT_NAME}/Defines.hpp.in
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DO/${DO_PROJECT_NAME}/Defines.hpp @ONLY)

# Set default build type to Release mode.
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Build type options are: Debug, Release")

# Set options.
option(DO_USE_FROM_SOURCE "Use DO-Sara libraries from source" ON)
option(DO_USE_VLD "Enable Visual Leak Detector for unit tests" OFF)
option(DO_BUILD_TESTS "Build unit tests for DO-Sara libraries" OFF)
option(DO_BUILD_SAMPLES "Build sample programs using DO-Sara libraries" OFF)
option(DO_BUILD_SHARED_LIBS "Build shared libraries for DO-Sara libraries"  OFF)

# Add DO-CV to the CMake module path.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(DO_Sara REQUIRED)
do_dissect_version(DO_${DO_PROJECT_NAME} ${DO_${DO_PROJECT_NAME}_VERSION})

# Group projects by category.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set installation settings.
include(do_installation_settings)

# Build third-party software (google test, FLANN,...)
add_subdirectory(third-party)

# Setup unit test projects
if (DO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif ()

# Setup unit test projects
if (DO_BUILD_SAMPLES)
  add_subdirectory(examples)
endif ()

# Make documentation
add_subdirectory(doc)

# To create Windows installer and Linux packages.
include(CPack)
