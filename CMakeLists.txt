cmake_minimum_required(VERSION 3.10)

project(Sara LANGUAGES C CXX)

# We are post-2020 now: let's all look to the future please.
# Activate C++20 globally
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Set the version major and minor manually.
set(DO_Sara_VERSION_MAJOR 1)
set(DO_Sara_VERSION_MINOR 7)

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
option(SARA_USE_QT6 "Switch to Qt6" OFF)
option(SARA_USE_HALIDE "Enable Halide" OFF)
option(SARA_BUILD_VIDEOIO "Build Sara's Video I/O module" OFF)
option(SARA_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(SARA_BUILD_TESTS "Build unit tests for DO-Sara libraries" OFF)
option(SARA_BUILD_SAMPLES "Build sample programs using DO-Sara libraries" OFF)
option(SARA_BUILD_SHARED_LIBS "Build shared libraries for DO-Sara libraries" OFF)
option(SARA_BUILD_DRAFTS "Build draft code" OFF)
option(SARA_SELF_CONTAINED_INSTALLATION
  "Install C++ and Python libraries in a single self contained directory" OFF)


# Set the following internal switches to build Sara.
set(SARA_USE_FROM_SOURCE TRUE CACHE INTERNAL "Build Sara libraries")
set(BUILD_SHARED_LIBS ${SARA_BUILD_SHARED_LIBS})

# Add Sara to the CMake module path.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(sara_create_asan_build_type)

# Import macros and configure Sara library version.
include(sara_macros)
sara_dissect_version()

# Use CCache if available to speed up builds.
include(sara_use_ccache)

if (CMAKE_SYSTEM_NAME STREQUAL Emscripten)
  set (CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
endif ()

# Use CUDA if available.
if (NOT CMAKE_SYSTEM_NAME STREQUAL Emscripten)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(TensorRT)
    find_package(OnnxRuntime)
  else ()
    message(STATUS "No CUDA support!")
  endif()
endif ()

if (CMAKE_SYSTEM_NAME STREQUAL iOS)
  # Find Boost package.
  set(Boost_INCLUDE_DIR
    /Users/david/GitHub/Apple-Boost-BuildScript/dist/boost.xcframework/Headers)
  set(Boost_LIBRARIES
    /Users/david/GitHub/Apple-Boost-BuildScript/dist/boost.xcframework/ios-arm64/libboost.a)
elseif (NOT CMAKE_SYSTEM_NAME STREQUAL Emscripten)
  find_package(TBB)

  # set(Boost_DEBUG ON)
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_MULTITHREADED ON)
  find_package(Boost
    COMPONENTS filesystem program_options serialization system
               unit_test_framework
    REQUIRED)

  find_package(OpenMP)

  find_package(HDF5 COMPONENTS CXX REQUIRED)

  find_package(JPEG REQUIRED)
  find_package(PNG  REQUIRED)
  find_package(TIFF REQUIRED)
  find_package(ZLIB REQUIRED)
  find_package(HEIF REQUIRED)
  find_package(WebP REQUIRED)

  if (SARA_USE_QT6)
    set (QT_VERSION 6)
    find_package(Qt${QT_VERSION} COMPONENTS Core Widgets OpenGL OpenGLWidgets REQUIRED)
  else (SARA_USE_QT6)
    set (QT_VERSION 5)
    find_package(Qt${QT_VERSION} COMPONENTS Core Widgets OpenGL OpenGLExtensions REQUIRED)
  endif ()

  # OpenGL/GLFW/ImGui
  find_package(OpenGL REQUIRED)
  if (NOT APPLE)
    find_package(GLEW REQUIRED)
  endif ()
  find_package(glfw3 REQUIRED)
  find_package(Freetype)

  find_package(OpenCL)

  find_package(Ceres REQUIRED)

  if (SARA_USE_HALIDE)
    include(sara_use_halide)
  endif ()
endif ()


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

  if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    find_package(XCTest REQUIRED)
  endif ()

  # Visual leak detector.
  if (WIN32 AND SARA_USE_VLD)
    find_package(VisualLeakDetector)
    include_directories(${VLD_INCLUDE_DIR})
    link_directories(${VLD_LINK_DIRECTORIES})
  endif ()
endif ()


# Build the libraries.
add_subdirectory(cpp)


if (NOT CMAKE_SYSTEM_NAME STREQUAL Emscripten)
  if (CMAKE_Swift_COMPILER)
    add_subdirectory(swift)
  endif ()

  if (SARA_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
  endif ()

  # To create Windows installer and Linux packages.
  include(CPack)
endif ()


# Make documentation
add_subdirectory(doc)
