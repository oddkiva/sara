cmake_minimum_required(VERSION 2.8.7)

project(Sara)

# Set the version major and minor manually.
set(DO_Sara_VERSION_MAJOR 1)
set(DO_Sara_VERSION_MINOR 1)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 OLD)
endif (POLICY CMP0020)

# Set default build type to Release mode.
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Build type options are: Debug, Release")

# Set options.
option(SARA_USE_FROM_SOURCE "Use DO-Sara libraries from source" ON)
option(SARA_USE_VLD "Enable Visual Leak Detector for unit tests" OFF)
option(SARA_BUILD_VIDEOIO "Build Sara's Video I/O module" OFF)
option(SARA_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(SARA_BUILD_TESTS "Build unit tests for DO-Sara libraries" OFF)
option(SARA_BUILD_SAMPLES "Build sample programs using DO-Sara libraries" OFF)
option(SARA_BUILD_SHARED_LIBS "Build shared libraries for DO-Sara libraries"  OFF)

set (SARA_INSTALL_DIR /usr CACHE STRING "Optional: Specify a directory path if want to install Sara C++ and Python libraries in a self-contained directory.")

set(BUILD_SHARED_LIBS ${SARA_BUILD_SHARED_LIBS})



# Configure templates.
configure_file(
  cmake/FindDO_Sara.cmake.in
  ${CMAKE_BINARY_DIR}/cmake/FindDO_Sara.cmake @ONLY)
file(COPY cmake
     DESTINATION ${CMAKE_BINARY_DIR})

# Add DO-Sara to the CMake module path.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/cmake)

# Set Sara library version.
include(sara_macros)
sara_dissect_version()



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
