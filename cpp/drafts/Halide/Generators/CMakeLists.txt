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

# Gaussian convolution V2 (teeny-tiny bit slower).
shakti_halide_library_v2(
  NAME shakti_gaussian_convolution_v2
  SRCS GaussianConvolutionV2Generator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

# Upscale function.
shakti_halide_library_v2(
  NAME shakti_scale_32f
  SRCS ResizeGenerators.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

# Enlarge function.
shakti_halide_library_v2(
  NAME shakti_reduce_32f
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


# Binary operators.
shakti_halide_library_v2(
  NAME shakti_subtract_32f
  SRCS BinaryOperatorGenerators.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# Local extrema operators.
shakti_halide_library_v2(
  NAME shakti_local_max_32f
  SRCS LocalExtremumGenerators.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

shakti_halide_library_v2(
  NAME shakti_local_scale_space_extremum_32f
  SRCS LocalExtremumGenerators.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

shakti_halide_library_v2(
  NAME shakti_refine_scale_space_extrema
  SRCS RefineExtremumGenerators.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# Gradient in polar coordinates in 2D.
shakti_halide_library_v2(
  NAME shakti_polar_gradient_2d_32f
  SRCS DifferentialGenerators.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

# Dominant gradient orientations.
shakti_halide_library_v2(
  NAME shakti_dominant_gradient_orientations
  SRCS DominantGradientOrientationsGenerator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


#  # SIFT descriptors.
#  shakti_halide_library_v2(
#    NAME shakti_sift_descriptors_32f
#    SRCS SIFTGenerator.cpp
#    HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


if (SARA_BUILD_SAMPLES)
  add_subdirectory(lessons)
  add_subdirectory(examples)
endif ()

if (SARA_BUILD_TESTS)
  add_subdirectory(test)
endif ()
