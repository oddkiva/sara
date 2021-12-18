find_package(DO_Sara COMPONENTS Core Graphics FeatureDescriptors REQUIRED)

file(GLOB test_SOURCE_FILES FILES *.cpp)

foreach (file ${test_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES ${DO_Sara_LIBRARIES} Halide::Halide Halide::Runtime)
  set_target_properties(${filename}
    PROPERTIES
    FOLDER "Tests/Shakti/Halide")
  target_link_options(${filename}
    PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO /FORCE /IGNORE:4006>)
endforeach ()

target_link_libraries(test_halide_cast
  PRIVATE
  shakti_cast_uint8_to_float_cpu)

target_link_libraries(test_halide_resize
  PRIVATE
  shakti_scale_32f_gpu
  shakti_reduce_32f_gpu
  shakti_enlarge_gpu)

target_link_libraries(test_halide_im2col
  PRIVATE
  shakti_im2col_32f_gpu)

target_link_libraries(test_halide_binary_operators
  PRIVATE
  shakti_subtract_32f_gpu)

target_link_libraries(test_halide_local_extrema
  PRIVATE
  shakti_local_max_32f_gpu
  shakti_local_scale_space_extremum_32f_gpu)

target_link_libraries(test_halide_dominant_gradient_orientations
  PRIVATE
  # Dominant gradient orientations.
  shakti_polar_gradient_2d_32f_gpu
  shakti_dominant_gradient_orientations_gpu)

target_link_libraries(test_halide_sift_descriptor
  PRIVATE
  shakti_polar_gradient_2d_32f_gpu
  shakti_sift_descriptor_gpu
  shakti_sift_descriptor_gpu_v2
  shakti_sift_descriptor_gpu_v3
  shakti_sift_descriptor_gpu_v4)
