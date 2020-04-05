include(sara_use_halide)

# N.B.: in Mac OS X, the genfiles directory is not formed correctly because of a
# bug in the CMake script in `Halide.cmake` is using the variable
# `CMAKE_CFG_INTDIR` incorrectly.
shakti_halide_library(shakti_halide_rgb_to_gray.cpp)
shakti_halide_library(shakti_halide_gray32f_to_rgb.cpp)
shakti_halide_gpu_library(shakti_halide_gaussian_blur.cpp)
shakti_halide_gpu_library(shakti_halide_enlarge.cpp)
shakti_halide_gpu_library(shakti_halide_cast_to_float.cpp)


shakti_halide_library_v2(
  NAME SeparableConvolution2d
  SRCS SeparableConvolution2dGenerator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

shakti_halide_library_v2(
  NAME GaussianConvolution
  SRCS GaussianConvolutionGenerator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS}
  DEPS SeparableConvolution2d.generator)


if (SARA_BUILD_SAMPLES)
  add_subdirectory(examples)
endif ()

if (SARA_BUILD_TESTS)
  add_subdirectory(test)
endif ()
