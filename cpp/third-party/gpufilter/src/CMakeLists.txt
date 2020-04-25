if (WIN32)
  cuda_add_library(gpufilter STATIC timer.cpp defs.cpp gpufilter.cu gpudefs.cu)
else ()
  add_library(gpufilter timer.cpp defs.cpp gpufilter.cu gpudefs.cu)
endif ()
set_target_properties(gpufilter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(gpufilter
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
target_compile_options(gpufilter
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --ptxas-options=-v
    -use_fast_math
    -arch=sm_${USE_SM}>
  PUBLIC
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:
    -Wno-unknown-pragmas>)

if (SARA_BUILD_SAMPLES)
  add_executable(ex1 example_sat1.cc)
  target_link_libraries(ex1 PRIVATE gpufilter)

  add_executable(ex2 example_sat2.cc)
  target_link_libraries(ex2 PRIVATE gpufilter)

  add_executable(ex3 example_sat3.cc)
  target_link_libraries(ex3 PRIVATE gpufilter ${CUDA_LIBRARIES})

  add_executable(ex4 example_r1.cc)
  target_link_libraries(ex4 PRIVATE gpufilter)

  add_executable(ex5 example_r2.cc)
  target_link_libraries(ex5 PRIVATE gpufilter)

  add_executable(ex6 example_r3.cc)
  target_link_libraries(ex6 PRIVATE gpufilter ${CUDA_LIBRARIES})

  add_executable(ex7 example_r4.cc)
  target_link_libraries(ex7 PRIVATE gpufilter)

  cuda_add_executable(ex8 example_r5.cu)
  target_link_libraries(ex8 PRIVATE gpufilter)

  add_executable(exB example_bspline.cc)
  target_link_libraries(exB PRIVATE gpufilter)

  add_executable(exG example_gauss.cc)
  target_link_libraries(exG PRIVATE gpufilter)

  add_executable(gaussian gaussian.cc)
  target_link_libraries(gaussian PRIVATE gpufilter)

  cuda_add_executable(alg5 run_alg5.cu)
  target_link_libraries(alg5 PRIVATE gpufilter)

  cuda_add_executable(alg5f4 run_alg5f4.cu)
  target_link_libraries(alg5f4 PRIVATE gpufilter)

  # include_directories(${OPENCV_DIR})
  # add_executable(recursive app_recursive.cc)
  # target_link_libraries(recursive cv highgui gpufilter)
endif ()
