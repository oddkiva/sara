file(GLOB SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../Utilities.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(DO_Shakti_Cuda_Utilities ${SRC_FILES})
set_target_properties(DO_Shakti_Cuda_Utilities
  PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  POSITION_INDEPENDENT_CODE ON
  FOLDER "Libraries/Shakti/Cuda")

# Propagate C++17 to any project linking against the library.
target_compile_features(DO_Shakti_Cuda_Utilities INTERFACE cxx_std_17)

# Set correct compile definitions when building the libraries.
 if (SARA_BUILD_SHARED_LIBS)
   target_compile_definitions(DO_Shakti_Cuda_Utilities
     PRIVATE DO_SHAKTI_EXPORTS)
 else ()
   target_compile_definitions(DO_Shakti_Cuda_Utilities
      PUBLIC DO_SHAKTI_STATIC)
endif ()

add_library(DO::Shakti::Cuda::Utilities ALIAS DO_Shakti_Cuda_Utilities)
