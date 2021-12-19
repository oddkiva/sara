find_package(DO_Sara
             COMPONENTS Core Graphics Geometry ImageIO SfM VideoIO REQUIRED)

# N.B.: with MSVC, this link helped to solve already defined symbol errors:
# https://github.com/halide/Halide/issues/740


add_executable(halide_aot_example halide_aot_example.cpp)
target_compile_definitions(halide_aot_example
  PRIVATE ${SARA_DEFINITIONS})
target_link_options(halide_aot_example
  PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO /FORCE /IGNORE:4006>)

target_link_libraries(halide_aot_example
  PRIVATE
  ${DO_Sara_LIBRARIES}
  Halide::Halide
  Halide::Runtime
  shakti_rgb8u_to_gray32f_cpu
  shakti_gray32f_to_rgb8u_cpu
  shakti_gaussian_convolution_gpu)
set_property(TARGET halide_aot_example PROPERTY FOLDER "Examples/Shakti/Halide")


add_executable(halide_resize_example
  halide_resize_example.cpp)
target_compile_definitions(halide_resize_example
  PRIVATE ${SARA_DEFINITIONS})
target_link_options(halide_resize_example
  PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO /FORCE /IGNORE:4006>)
target_link_libraries(halide_resize_example
  PRIVATE
  ${DO_Sara_LIBRARIES}
  Halide::Halide
  Halide::Runtime
  shakti_rgb8u_to_gray32f_cpu
  shakti_gray32f_to_rgb8u_cpu
  shakti_reduce_32f_gpu)
set_property(TARGET halide_resize_example
  PROPERTY
  FOLDER "Examples/Shakti/Halide")


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


if (CMAKE_CUDA_COMPILER)
  cuda_add_executable(halide_sift_example halide_sift_example.cpp)
else ()
  add_executable(halide_sift_example halide_sift_example.cpp)
endif ()
target_compile_definitions(halide_sift_example
  PRIVATE
  ${SARA_DEFINITIONS}
  $<$<BOOL:${NvidiaVideoCodec_ROOT}>:USE_SHAKTI_CUDA_VIDEOIO>)
target_link_options(halide_sift_example
  PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO /FORCE /IGNORE:4006>)
target_link_libraries(halide_sift_example
  PRIVATE
  Halide::Halide
  Halide::Runtime
  ${DO_Sara_LIBRARIES}
  ${HALIDE_SIFT_LIBRARIES}
  $<$<BOOL:${CMAKE_CUDA_COMPILER}>:${CUDA_CUDA_LIBRARY}>
  $<$<BOOL:${NvidiaVideoCodec_ROOT}>:DO::Shakti::VideoIO>)

set_property(TARGET halide_sift_example
  PROPERTY
  FOLDER "Examples/Shakti/Halide")

add_executable(halide_sift_octave_example
  halide_sift_octave_example.cpp)
target_compile_definitions(halide_sift_octave_example
  PRIVATE ${SARA_DEFINITIONS})
target_link_options(halide_sift_octave_example
  PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO /FORCE /IGNORE:4006>)
target_link_libraries(halide_sift_octave_example
  PRIVATE
  Halide::Halide
  Halide::Runtime
  ${DO_Sara_LIBRARIES}
  ${HALIDE_SIFT_LIBRARIES})
set_property(TARGET halide_sift_octave_example
  PROPERTY
  FOLDER "Examples/Shakti/Halide")

add_executable(halide_sift_pyramid_example
  halide_sift_pyramid_example.cpp)
target_compile_definitions(halide_sift_pyramid_example
  PRIVATE ${SARA_DEFINITIONS}
          $<$<BOOL:${NvidiaVideoCodec_ROOT}>:USE_SHAKTI_CUDA_VIDEOIO>)
target_link_options(halide_sift_pyramid_example
  PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO /FORCE /IGNORE:4006>)
target_link_libraries(halide_sift_pyramid_example
  PRIVATE
  Halide::Halide
  Halide::Runtime
  ${DO_Sara_LIBRARIES}
  ${HALIDE_SIFT_LIBRARIES}
  $<$<BOOL:${CMAKE_CUDA_COMPILER}>:${CUDA_CUDA_LIBRARY}>
  $<$<BOOL:${NvidiaVideoCodec_ROOT}>:DO::Shakti::VideoIO>)
set_property(TARGET halide_sift_pyramid_example
  PROPERTY
  FOLDER "Examples/Shakti/Halide")

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


add_executable(halide_edge_detection_example
  halide_edge_detection_example.cpp)
target_compile_definitions(halide_edge_detection_example
  PRIVATE ${SARA_DEFINITIONS})
target_link_options(halide_edge_detection_example
  PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO /FORCE /IGNORE:4006>)
target_link_libraries(halide_edge_detection_example
  PRIVATE
  Halide::Halide
  Halide::Runtime
  shakti_rgb8u_to_gray32f_cpu
  shakti_gaussian_convolution_gpu
  shakti_scale_32f_gpu
  shakti_polar_gradient_2d_32f_gpu_v2
  ${DO_Sara_LIBRARIES})
set_property(TARGET halide_edge_detection_example
  PROPERTY
  FOLDER "Examples/Shakti/Halide")
