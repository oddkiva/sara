# N.B.: in Mac OS X, the genfiles directory is not formed correctly because of a
# bug in the CMake script in `Halide.cmake` is using the variable
# `CMAKE_CFG_INTDIR` incorrectly.

# ------------------------------------------------------------------------------
# Color conversions.
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_rgb8u_to_gray32f_cpu
  SRCS FromRgb8uToGray32f.cpp)

shakti_halide_library_v2(
  NAME shakti_bgra8u_to_gray32f_cpu
  SRCS FromBgra8uToGray32f.cpp)

shakti_halide_library_v2(
  NAME shakti_gray32f_to_rgb8u_cpu
  SRCS FromGray32fToRgb8u.cpp)

shakti_halide_library_v2(
  NAME shakti_cast_uint8_to_float_cpu
  SRCS CastUint8ToFloat.cpp)


# ------------------------------------------------------------------------------
# Separable convolution 2D
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_separable_convolution_2d_cpu
  SRCS SeparableConvolution2dGenerator.cpp)

shakti_halide_library_v2(
  NAME shakti_separable_convolution_2d_gpu
  SRCS SeparableConvolution2dGenerator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Gaussian convolution
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_gaussian_convolution_cpu
  SRCS GaussianConvolutionGenerator.cpp)

shakti_halide_library_v2(
  NAME shakti_gaussian_convolution_gpu
  SRCS GaussianConvolutionGenerator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Upscale function.
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_scale_32f_cpu
  SRCS ResizeGenerators.cpp)

shakti_halide_library_v2(
  NAME shakti_scale_32f_gpu
  SRCS ResizeGenerators.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Reduce function
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_reduce_32f_cpu
  SRCS ResizeGenerators.cpp)

shakti_halide_library_v2(
  NAME shakti_reduce_32f_gpu
  SRCS ResizeGenerators.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Enlarge function
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_enlarge_cpu
  SRCS ResizeGenerators.cpp)

shakti_halide_library_v2(
  NAME shakti_enlarge_gpu
  SRCS ResizeGenerators.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Im2Col
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_im2col_32f_gpu
  SRCS Im2ColGenerator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Batched convolution.
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_convolve_batch_32f_cpu
  SRCS BatchedConvolutionGenerator.cpp)

shakti_halide_library_v2(
  NAME shakti_convolve_batch_32f_gpu
  SRCS BatchedConvolutionGenerator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Binary operators.
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_subtract_32f_cpu
  SRCS BinaryOperatorGenerators.cpp)

shakti_halide_library_v2(
  NAME shakti_subtract_32f_gpu
  SRCS BinaryOperatorGenerators.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Local extrema operators.
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_local_scale_space_extremum_32f_gpu_v2
  SRCS LocalExtremumGeneratorsV2.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

shakti_halide_library_v2(
  NAME shakti_local_scale_space_extremum_32f_gpu_v3
  SRCS LocalExtremumGeneratorsV2.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Scale-space Extremum refinement (using a 2nd-order Taylor expansion)
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_refine_scale_space_extrema_gpu_v2
  SRCS RefineExtremumGeneratorsV2.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

shakti_halide_library_v2(
  NAME shakti_refine_scale_space_extrema_gpu_v3
  SRCS RefineExtremumGeneratorsV2.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Forward difference operator.
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_forward_difference_32f_gpu
  SRCS DifferentialGeneratorsV2.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Gradients in Cartesian coordinates
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_gradient_2d_32f_cpu
  SRCS DifferentialGeneratorsV2.cpp)

shakti_halide_library_v2(
  NAME shakti_gradient_2d_32f_gpu_v2
  SRCS DifferentialGeneratorsV2.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Gradients in polar coordinates
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_polar_gradient_2d_32f_cpu
  SRCS DifferentialGeneratorsV2.cpp)

shakti_halide_library_v2(
  NAME shakti_polar_gradient_2d_32f_gpu_v2
  SRCS DifferentialGeneratorsV2.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Dominant gradient orientations
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_dominant_gradient_orientations_gpu_v2
  SRCS DominantGradientOrientationsGeneratorV2.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

shakti_halide_library_v2(
  NAME shakti_dominant_gradient_orientations_gpu_v3
  SRCS DominantGradientOrientationsGeneratorV2.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# SIFT descriptors
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_sift_descriptor_gpu
  SRCS SIFTGenerator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

shakti_halide_library_v2(
  NAME shakti_sift_descriptor_gpu_v2
  SRCS SIFTGeneratorV2.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

shakti_halide_library_v2(
  NAME shakti_sift_descriptor_gpu_v3
  SRCS SIFTGeneratorV3.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

shakti_halide_library_v2(
  NAME shakti_sift_descriptor_gpu_v4
  SRCS SIFTGeneratorV4.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

shakti_halide_library_v2(
  NAME shakti_sift_descriptor_gpu_v5
  SRCS SIFTGeneratorV5.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

# ------------------------------------------------------------------------------
# Stream compaction to populate extrema
# ------------------------------------------------------------------------------
shakti_halide_library_v2(
  NAME shakti_count_extrema_gpu
  SRCS CountGenerator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

shakti_halide_library_v2(
  NAME shakti_stream_compaction_gpu
  SRCS StreamCompactionGenerator.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})


# ------------------------------------------------------------------------------
# Examples
# ------------------------------------------------------------------------------
if (SARA_BUILD_SAMPLES)
  add_subdirectory(examples)
endif ()


# ------------------------------------------------------------------------------
# Examples
# ------------------------------------------------------------------------------
if (SARA_BUILD_TESTS)
  add_subdirectory(test)
endif ()
