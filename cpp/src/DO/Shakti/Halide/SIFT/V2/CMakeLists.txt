find_package(
  DO_Sara
  COMPONENTS Core
  REQUIRED)

set(HALIDE_SIFT_BLOCK_LIBRARIES_V2
    # Resize operations.
    shakti_enlarge_gpu
    shakti_reduce_32f_gpu
    shakti_scale_32f_gpu
    # Gaussian pyramid.
    shakti_separable_convolution_2d_gpu
    shakti_gaussian_convolution_gpu
    # DoG pyramid.
    shakti_subtract_32f_gpu
    # Scale-space extrema.
    shakti_local_max_32f_gpu
    shakti_local_scale_space_extremum_32f_gpu_v2
    # Refine extrema.
    shakti_refine_scale_space_extrema_gpu_v2
    # Dominant gradient orientations.
    shakti_polar_gradient_2d_32f_gpu_v2
    shakti_dominant_gradient_orientations_gpu_v2
    # SIFT descriptor.
    # shakti_sift_descriptor_gpu
    # shakti_sift_descriptor_gpu_v2
    # shakti_sift_descriptor_gpu_v3
    # shakti_sift_descriptor_gpu_v4
)

add_library(DO_Shakti_Halide_Sift_gpu_v2 Pipeline.hpp Pipeline.cpp)
target_link_libraries(
  DO_Shakti_Halide_Sift_gpu_v2
  PUBLIC Halide::Halide #
         Halide::Runtime #
  PRIVATE DO::Sara::Core #
          ${HALIDE_SIFT_BLOCK_LIBRARIES_V2})

add_subdirectory(examples)
