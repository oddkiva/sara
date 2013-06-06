cmake_minimum_required(VERSION 2.8.7)

project(DO)
set(DO_VERSION 1.0.0)

# Configure DO++: put the versions.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindDO.cmake.in
               ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindDO.cmake @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/DO/Defines.hpp.in
               ${CMAKE_CURRENT_SOURCE_DIR}/src/DO/Defines.hpp @ONLY)

# Set options.
option(DO_USE_FROM_SOURCE   "Use DO++ libraries from source" ON)
option(DO_USE_STATIC_LIBS   "Use DO++ static libraries" OFF)
option(DO_BUILD_UNIT_TESTS  "Build unit tests for DO++ libraries" ON)
option(DO_BUILD_SHARED_LIBS "Build shared libraries for DO++ libraries" OFF)

# Go!
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(DO REQUIRED)
do_dissect_version()
do_get_os_info()

# Build third-party software (google test, FLANN,...)
add_subdirectory(third-party)

# Install packages
include(DOInstall)

# Setup unit test projects
if (DO_BUILD_UNIT_TESTS)
  add_subdirectory(test)
  enable_testing()
endif ()

# Make documentation
add_subdirectory(doc)

# Make packages
include(DOMakePackages)