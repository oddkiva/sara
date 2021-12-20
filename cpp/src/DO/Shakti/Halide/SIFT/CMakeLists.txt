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

