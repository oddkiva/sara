cuda_compile(CUDA_SOURCES gpudefs.cu)
cuda_add_library(gpufilter timer.cpp defs.cpp ${CUDA_SOURCES})
target_include_directories(gpufilter
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)


if (SARA_BUILD_SAMPLES)
  add_executable(ex1 example_sat1.cc)
  target_link_libraries(ex1 gpufilter)

  cuda_add_executable(ex2 example_sat2.cc gpufilter.cu)
  target_link_libraries(ex2 gpufilter)

  cuda_add_executable(ex3 example_sat3.cc gpufilter.cu)
  target_link_libraries(ex3 gpufilter)

  add_executable(ex4 example_r1.cc)
  target_link_libraries(ex4 gpufilter)

  cuda_add_executable(ex5 example_r2.cc gpufilter.cu)
  target_link_libraries(ex5 gpufilter)

  cuda_add_executable(ex6 example_r3.cc gpufilter.cu)
  target_link_libraries(ex6 gpufilter)

  cuda_add_executable(ex7 example_r4.cc gpufilter.cu)
  target_link_libraries(ex7 gpufilter)

  cuda_add_executable(ex8 example_r5.cu gpufilter.cu)
  target_link_libraries(ex8 gpufilter)

  cuda_add_executable(exB example_bspline.cc gpufilter.cu)
  target_link_libraries(exB gpufilter)

  cuda_add_executable(exG example_gauss.cc gpufilter.cu)
  target_link_libraries(exG gpufilter)

  add_executable(gaussian gaussian.cc)
  target_link_libraries(gaussian gpufilter)

  cuda_add_executable(alg5 run_alg5.cu)
  target_link_libraries(alg5 gpufilter)

  cuda_add_executable(alg5f4 run_alg5f4.cu)
  target_link_libraries(alg5f4 gpufilter)

  #include_directories(${OPENCV_DIR})

  #cuda_add_executable(recursive app_recursive.cc)
  #target_link_libraries(recursive cv highgui gpufilter)
endif ()

