include(sara_use_halide)

# N.B.: in Mac OS X, the genfiles directory is not formed correctly because of a
# bug in the CMake script in `Halide.cmake` is using the variable
# `CMAKE_CFG_INTDIR` incorrectly.

# Color conversions.
shakti_halide_library(shakti_halide_rgb_to_gray.cpp)
shakti_halide_library(shakti_halide_gray32f_to_rgb.cpp)
shakti_halide_gpu_library(shakti_halide_cast_to_float.cpp)

# Fastest gaussian convolution at the moment
shakti_halide_gpu_library(shakti_halide_gaussian_blur.cpp)


# Separable convolution 2D
shakti_halide_library_v2(
  NAME shakti_separable_convolution_2d
  SRCS SeparableConvolution2dGenerator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

# Gaussian convolution (slower implementation).
shakti_halide_library_v2(
  NAME shakti_gaussian_convolution
  SRCS GaussianConvolutionGenerator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS}
  DEPS shakti_separable_convolution_2d.generator)

# Gaussian convolution V2
shakti_halide_library_v2(
  NAME shakti_gaussian_convolution_v2
  SRCS GaussianConvolutionV2Generator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

# Upscale function.
shakti_halide_library_v2(
  NAME shakti_upscale_32f
  SRCS ResizeGenerators.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

# Enlarge function.
shakti_halide_library_v2(
  NAME shakti_enlarge
  SRCS ResizeGenerators.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# Im2Col.
shakti_halide_library_v2(
  NAME shakti_im2col_32f
  SRCS Im2ColGenerator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


if (SARA_BUILD_SAMPLES)
  add_subdirectory(examples)
endif ()

if (SARA_BUILD_TESTS)
  add_subdirectory(test)
endif ()
