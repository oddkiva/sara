find_package(DO_Sara
             COMPONENTS Core Graphics Geometry ImageIO SfM VideoIO REQUIRED)

# N.B.: with MSVC, this link helped to solve already defined symbol errors:
# https://github.com/halide/Halide/issues/740


set(HALIDE_SIFT_LIBRARIES
  shakti_rgb8u_to_gray32f_cpu
  shakti_bgra8u_to_gray32f_cpu
  shakti_gray32f_to_rgb8u_cpu
  # Resize operations.
  shakti_enlarge_gpu
  shakti_reduce_32f_gpu
  shakti_scale_32f_gpu
  # Gaussian pyramid.
  shakti_gaussian_convolution_gpu
  shakti_convolve_batch_32f_gpu
  # DoG pyramid.
  shakti_subtract_32f_gpu
  shakti_forward_difference_32f_gpu
  # Scale-space extrema.
  shakti_local_max_32f_gpu
  shakti_local_scale_space_extremum_32f_gpu
  shakti_local_scale_space_extremum_32f_gpu_v2
  shakti_local_scale_space_extremum_32f_gpu_v3
  # Refine extrema.
  shakti_refine_scale_space_extrema_gpu
  shakti_refine_scale_space_extrema_gpu_v2
  shakti_refine_scale_space_extrema_gpu_v3
  # Dominant gradient orientations.
  shakti_polar_gradient_2d_32f_gpu
  shakti_polar_gradient_2d_32f_gpu_v2
  shakti_dominant_gradient_orientations_gpu
  shakti_dominant_gradient_orientations_gpu_v2
  shakti_dominant_gradient_orientations_gpu_v3
  # SIFT descriptor.
  shakti_sift_descriptor_gpu
  shakti_sift_descriptor_gpu_v2
  shakti_sift_descriptor_gpu_v3
  shakti_sift_descriptor_gpu_v4
  # Stream compaction
  shakti_count_extrema_gpu
  shakti_stream_compaction_gpu
)


add_executable(halide_batched_sift_octave_example
  halide_batched_sift_octave_example.cpp)
target_compile_definitions(halide_batched_sift_octave_example
  PRIVATE ${SARA_DEFINITIONS})
target_link_options(halide_batched_sift_octave_example
  PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO /FORCE /IGNORE:4006>)
target_link_libraries(halide_batched_sift_octave_example
  PRIVATE
  Halide::Halide
  Halide::Runtime
  ${DO_Sara_LIBRARIES}
  ${HALIDE_SIFT_LIBRARIES})
set_property(TARGET halide_batched_sift_octave_example
  PROPERTY
  FOLDER "Examples/Shakti/Halide")
