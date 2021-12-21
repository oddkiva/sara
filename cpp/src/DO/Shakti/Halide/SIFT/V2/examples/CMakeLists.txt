find_package(DO_Sara
             COMPONENTS Core Graphics Geometry ImageIO SfM VideoIO REQUIRED)

# N.B.: with MSVC, this link helped to solve already defined symbol errors:
# https://github.com/halide/Halide/issues/740

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
  DO_Shakti_Halide_Sift_gpu_v2
  shakti_bgra8u_to_gray32f_cpu
  shakti_rgb8u_to_gray32f_cpu)
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
  ${DO_Sara_LIBRARIES}
  ${Boost_LIBRARIES}
  DO_Shakti_Halide_Sift_gpu_v2
  shakti_bgra8u_to_gray32f_cpu
  shakti_rgb8u_to_gray32f_cpu
  $<$<BOOL:${CMAKE_CUDA_COMPILER}>:${CUDA_CUDA_LIBRARY}>
  $<$<BOOL:${NvidiaVideoCodec_ROOT}>:DO::Shakti::VideoIO>)
set_property(TARGET halide_sift_pyramid_example
  PROPERTY
  FOLDER "Examples/Shakti/Halide")
