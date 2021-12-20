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
