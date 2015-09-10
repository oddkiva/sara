cmake_minimum_required(VERSION 2.8.7)

set(DO_PROJECT_NAME Sara CACHE STRING "DO Project")
project(${DO_PROJECT_NAME})
set(DO_${DO_PROJECT_NAME}_VERSION 1.1.1)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 OLD)
endif (POLICY CMP0020)

# Set Sara Library version.
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

set(BUILD_SHARED_LIBS ${DO_BUILD_SHARED_LIBS})

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

  if (CMAKE_COMPILER_IS_GNUCXX)
    find_program(LCOV_PATH lcov)
    find_program(GCOV_PATH gcov-${GCC_MAJOR}.${GCC_MINOR})
    find_program(GENHTML_PATH genhtml)

    if (LCOV_PATH AND GCOV_PATH AND GENHTML_PATH)
      add_custom_target(coverage
        COMMAND ${LCOV_PATH} --gcov-tool=${GCOV_PATH} --compat-libtool
                     --directory=${CMAKE_BINARY_DIR}
                     --base-directory=${CMAKE_CURRENT_SOURCE_DIR}/src
                     --capture
                     --output-file=coverage.info
        COMMAND ${LCOV_PATH} --remove coverage.info 'usr/*'
                     --output-file coverage.info
        COMMAND ${LCOV_PATH} --remove coverage.info 'third-party/*'
                     --output-file coverage.info
        COMMAND ${LCOV_PATH} --remove coverage.info 'test/*'
                     --output-file coverage.info
        COMMAND ${LCOV_PATH} --remove coverage.info '${CMAKE_BINARY_DIR}/moc_*'
                     --output-file coverage.info
        COMMAND ${GENHTML_PATH} coverage.info
                  --output-directory=${CMAKE_BINARY_DIR}/coverage
      )
    endif ()
  endif ()
endif ()

# Setup unit test projects
if (DO_BUILD_SAMPLES)
  add_subdirectory(examples)
endif ()

# Make documentation
add_subdirectory(doc)

# To create Windows installer and Linux packages.
include(CPack)
