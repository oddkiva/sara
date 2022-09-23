add_library(
  DO_Sara_TensorRT #
  DarknetParser.hpp #
  Helpers.hpp #
  IO.hpp #
  Yolo.cpp
  Yolo.hpp
  Yolo.cu)
add_library(DO::Sara::TensorRT ALIAS DO_Sara_TensorRT)
target_link_libraries(
  DO_Sara_TensorRT
  PUBLIC $<$<PLATFORM_ID:Linux>:CUDA::cudart> #
         ${TensorRT_LIBRARIES} #
         DO::Sara::Core)

if(SARA_BUILD_TESTS)
  add_subdirectory(test)
endif()
