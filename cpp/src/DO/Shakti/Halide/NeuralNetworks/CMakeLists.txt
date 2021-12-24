shakti_halide_library_v2(
  NAME shakti_nn_convolution_cpu
  SRCS Convolution.cpp)

shakti_halide_library_v2(
  NAME shakti_nn_convolution_cpu
  SRCS Convolution.cpp
  HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})
