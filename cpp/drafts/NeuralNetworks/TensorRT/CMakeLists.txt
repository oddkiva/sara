if(NOT CMAKE_CUDA_COMPILER OR NOT TensorRT_FOUND)
  return()
endif ()

add_library(
  DO_Sara_TensorRT #
  Helpers.hpp #
  IO.hpp #
  IO.cpp #
  InferenceExecutor.hpp #
  InferenceExecutor.cpp #
  DarknetParser.hpp #
  DarknetParser.cpp #
  Yolo.cpp
  Yolo.hpp
  YoloImpl.hpp
  YoloImpl.cu)
add_library(DO::Sara::TensorRT ALIAS DO_Sara_TensorRT)
target_link_libraries(
  DO_Sara_TensorRT
  PUBLIC $<$<PLATFORM_ID:Linux>:CUDA::cudart> #
         TensorRT::TensorRT #
         DO::Sara::Darknet)

add_subdirectory(examples)

if(SARA_BUILD_TESTS)
  add_subdirectory(test)
endif()
