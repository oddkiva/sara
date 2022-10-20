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
if (SARA_BUILD_SHARED_LIBS)
  target_compile_definitions(DO_Sara_TensorRT PRIVATE DO_SARA_EXPORTS)
else ()
  target_compile_definitions(DO_Sara_TensorRT PUBLIC DO_SARA_STATIC)
endif ()

target_link_libraries(
  DO_Sara_TensorRT
  PUBLIC $<$<PLATFORM_ID:Linux>:CUDA::cudart> #
         TensorRT::TensorRT #
         DO::Shakti::Cuda::Utilities
         DO::Sara::Darknet)
set_property(TARGET DO_Sara_TensorRT PROPERTY FOLDER "Libraries/Sara")

add_subdirectory(examples)

if(SARA_BUILD_TESTS)
  add_subdirectory(test)
endif()
