project(Shakti CXX)


# Set the version major and minor manually.
set(DO_Shakti_VERSION_MAJOR 0)
set(DO_Shakti_VERSION_MINOR 1)

# List the build options.
option(SHAKTI_BUILD_PYTHON_BINDINGS "Build Python bindings"
  ${SARA_BUILD_PYTHON_BINDINGS})
option(SHAKTI_BUILD_SAMPLES "Build samples in Shakti libraries"
  ${SARA_BUILD_SAMPLES})
option(SHAKTI_BUILD_TESTS "Build unit tests for Shakti libraries"
  ${SARA_BUILD_TESTS})
option(SHAKTI_SELF_CONTAINED_INSTALLATION
  "Install C++ and Python libraries in a single self-contained directory"
  ${SARA_SELF_CONTAINED_INSTALLATION})
option(SHAKTI_BUILD_SHARED_LIBS "Build shared libraries for DO-Shakti libraries"
  ${SARA_BUILD_SHARED_LIBS})

include(shakti_macros)

# Because we use the old CMake way to use CUDA.
include_directories(
  ${CMAKE_SOURCE_DIR}/cpp/src
  ${CMAKE_SOURCE_DIR}/cpp/third-party)

# Use custom CMake macros.
set(DO_Shakti_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DO_Shakti_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
shakti_dissect_version()

# Group projects by category.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (CMAKE_CUDA_COMPILER)
  sara_message("Building DO::Shakti CUDA Libraries")
  add_subdirectory(Cuda)
endif ()
if (SARA_USE_HALIDE)
  add_subdirectory(Halide)
endif ()
