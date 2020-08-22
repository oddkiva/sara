find_package(OpenMP)
find_package(DO_Sara COMPONENTS Core Graphics ImageIO SfM VideoIO REQUIRED)

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
  Halide
  shakti_halide_rgb_to_gray
  shakti_halide_gray32f_to_rgb
  shakti_halide_gaussian_blur
  shakti_gaussian_convolution_v2)
set_property(TARGET halide_aot_example PROPERTY FOLDER "Tutorials/Halide")


add_executable(halide_gaussian_convolution_example
  halide_gaussian_convolution_example.cpp)
target_compile_definitions(halide_gaussian_convolution_example
  PRIVATE ${SARA_DEFINITIONS})
target_link_libraries(halide_gaussian_convolution_example
  PRIVATE
  ${DO_Sara_LIBRARIES}
  Halide
  shakti_gaussian_convolution_v2
  shakti_halide_gaussian_blur)
set_property(TARGET halide_gaussian_convolution_example
  PROPERTY
  FOLDER "Tutorials/Halide")

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
  Halide
  shakti_halide_rgb_to_gray
  shakti_halide_gray32f_to_rgb
  shakti_reduce_32f)
set_property(TARGET halide_resize_example
  PROPERTY
  FOLDER "Tutorials/Halide")

add_executable(halide_sift_example
  halide_sift_example.cpp)
target_compile_definitions(halide_sift_example
  PRIVATE ${SARA_DEFINITIONS})
target_link_options(halide_sift_example
  PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO /FORCE /IGNORE:4006>)
target_link_libraries(halide_sift_example
  PRIVATE
  ${DO_Sara_LIBRARIES}
  Halide
  shakti_halide_rgb_to_gray
  shakti_halide_gray32f_to_rgb
  # Scale-space generation.
  shakti_enlarge
  shakti_reduce_32f
  shakti_scale_32f
  # Gaussian pyramid.
  shakti_gaussian_convolution_v2
  # DoG pyramid.
  shakti_subtract_32f
  # Scale-space extrema.
  shakti_local_max_32f
  shakti_local_scale_space_extremum_32f
  # Refine extrema.
  shakti_refine_scale_space_extrema
  # Dominant gradient orientations.
  shakti_polar_gradient_2d_32f
  shakti_dominant_gradient_orientations
  # SIFT descriptor.
  shakti_sift_descriptor
  shakti_sift_descriptor_v2
  shakti_sift_descriptor_v3
  shakti_sift_descriptor_v4
  )
set_property(TARGET halide_sift_example
  PROPERTY
  FOLDER "Tutorials/Halide")

add_executable(halide_sift_octave_example
  halide_sift_octave_example.cpp)
target_compile_definitions(halide_sift_octave_example
  PRIVATE ${SARA_DEFINITIONS})
target_link_options(halide_sift_octave_example
  PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO /FORCE /IGNORE:4006>)
target_link_libraries(halide_sift_octave_example
  PRIVATE
  ${DO_Sara_LIBRARIES}
  Halide
  shakti_halide_rgb_to_gray
  shakti_halide_gray32f_to_rgb
  # Scale-space generation.
  shakti_enlarge
  shakti_reduce_32f
  shakti_scale_32f
  # Gaussian pyramid.
  shakti_gaussian_convolution_v2
  # DoG pyramid.
  shakti_subtract_32f
  # Scale-space extrema.
  shakti_local_max_32f
  shakti_local_scale_space_extremum_32f
  shakti_local_scale_space_extremum_32f_v2
  # Refine extrema.
  shakti_refine_scale_space_extrema
  shakti_refine_scale_space_extrema_v2
  # Dominant gradient orientations.
  shakti_polar_gradient_2d_32f
  shakti_polar_gradient_2d_32f_v2
  shakti_dominant_gradient_orientations
  shakti_dominant_gradient_orientations_v2
  # SIFT descriptor.
  shakti_sift_descriptor
  shakti_sift_descriptor_v2
  shakti_sift_descriptor_v3
  shakti_sift_descriptor_v4
  )
set_property(TARGET halide_sift_octave_example
  PROPERTY
  FOLDER "Tutorials/Halide")

add_executable(halide_sift_pyramid_example
  halide_sift_pyramid_example.cpp)
target_compile_definitions(halide_sift_pyramid_example
  PRIVATE ${SARA_DEFINITIONS})
target_link_options(halide_sift_pyramid_example
  PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO /FORCE /IGNORE:4006>)
target_link_libraries(halide_sift_pyramid_example
  PRIVATE
  ${DO_Sara_LIBRARIES}
  Halide
  shakti_halide_rgb_to_gray
  shakti_halide_gray32f_to_rgb
  # Scale-space generation.
  shakti_enlarge
  shakti_reduce_32f
  shakti_scale_32f
  # Gaussian pyramid.
  shakti_gaussian_convolution_v2
  # DoG pyramid.
  shakti_subtract_32f
  # Scale-space extrema.
  shakti_local_max_32f
  shakti_local_scale_space_extremum_32f
  shakti_local_scale_space_extremum_32f_v2
  # Refine extrema.
  shakti_refine_scale_space_extrema
  shakti_refine_scale_space_extrema_v2
  # Dominant gradient orientations.
  shakti_polar_gradient_2d_32f
  shakti_polar_gradient_2d_32f_v2
  shakti_dominant_gradient_orientations
  shakti_dominant_gradient_orientations_v2
  # SIFT descriptor.
  shakti_sift_descriptor
  shakti_sift_descriptor_v2
  shakti_sift_descriptor_v3
  shakti_sift_descriptor_v4
  )
set_property(TARGET halide_sift_pyramid_example
  PROPERTY
  FOLDER "Tutorials/Halide")

add_executable(halide_batched_convolution_example
  halide_batched_convolution_example.cpp)
target_compile_definitions(halide_batched_convolution_example
  PRIVATE ${SARA_DEFINITIONS})
target_link_options(halide_batched_convolution_example
  PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO /FORCE /IGNORE:4006>)
target_link_libraries(halide_batched_convolution_example
  PRIVATE
  ${DO_Sara_LIBRARIES}
  Halide
  shakti_halide_rgb_to_gray
  shakti_halide_gray32f_to_rgb
  shakti_convolve_batch_32f
  shakti_forward_difference_32f
  )
set_property(TARGET halide_batched_convolution_example
  PROPERTY
  FOLDER "Tutorials/Halide")
