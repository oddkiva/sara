find_package(DO_Sara COMPONENTS Core Graphics FeatureDescriptors REQUIRED)

file(GLOB test_SOURCE_FILES FILES *.cpp)

foreach (file ${test_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES ${DO_Sara_LIBRARIES} Halide
    FOLDER "DRAFTS/Halide")
  set_target_properties(${filename}
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES)
endforeach ()

target_link_libraries(test_halide_cast
  PRIVATE
  shakti_halide_cast_to_float)

target_link_libraries(test_halide_resize
  PRIVATE
  shakti_scale_32f
  shakti_reduce_32f
  shakti_enlarge)

target_link_libraries(test_halide_im2col
  PRIVATE
  shakti_im2col_32f)

target_link_libraries(test_halide_binary_operators
  PRIVATE
  shakti_subtract_32f)

target_link_libraries(test_halide_local_extrema
  PRIVATE
  shakti_local_max_32f
  shakti_local_scale_space_extremum_32f)

target_link_libraries(test_halide_dominant_gradient_orientations
  PRIVATE
  # Dominant gradient orientations.
  shakti_polar_gradient_2d_32f
  shakti_dominant_gradient_orientations)

target_link_libraries(test_halide_sift_descriptor
  PRIVATE
  shakti_polar_gradient_2d_32f
  shakti_sift_descriptor
  shakti_sift_descriptor_v2)
