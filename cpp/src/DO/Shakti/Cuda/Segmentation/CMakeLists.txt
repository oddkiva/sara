file(GLOB_RECURSE SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../Segmentation.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(DO_Shakti_Cuda_Segmentation ${SRC_FILES})
set_target_properties(DO_Shakti_Cuda_Segmentation
  PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  POSITION_INDEPENDENT_CODE ON
  FOLDER "Libraries/Shakti/Cuda")

target_compile_features(DO_Shakti_Cuda_Segmentation INTERFACE cxx_std_17)

# Set correct compile definitions when building the libraries.
 if (SARA_BUILD_SHARED_LIBS)
   target_compile_definitions(DO_Shakti_Cuda_Segmentation
     PRIVATE DO_SHAKTI_EXPORTS)
 else ()
   target_compile_definitions(DO_Shakti_Cuda_Segmentation
      PUBLIC DO_SHAKTI_STATIC)
endif ()

target_link_libraries(DO_Shakti_Cuda_Segmentation
  PUBLIC
  DO::Shakti::Cuda::MultiArray
  DO::Shakti::Cuda::Utilities)

add_library(DO::Shakti::Cuda::Segmentation ALIAS DO_Shakti_Cuda_Segmentation)
