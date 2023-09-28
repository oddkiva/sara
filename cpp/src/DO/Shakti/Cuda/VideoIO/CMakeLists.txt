add_library(DO_Shakti_Cuda_VideoIO
  ${CMAKE_CURRENT_SOURCE_DIR}/../VideoIO.hpp
  VideoStream.hpp VideoStream.cu)

target_link_libraries(DO_Shakti_Cuda_VideoIO
  PUBLIC
  CUDA::cuda_driver
  nvidia::VideoCodecHelpers
  ${FFMPEG_LIBRARIES}
  DO::Sara::Core)

 set_target_properties(DO_Shakti_Cuda_VideoIO
  PROPERTIES
  CXX_STANDARD_REQUIRED YES
  POSITION_INDEPENDENT_CODE ON
  FOLDER "Libraries/Shakti/Cuda")

# Set correct compile definitions when building the libraries.
 if (SARA_BUILD_SHARED_LIBS)
   target_compile_definitions(DO_Shakti_Cuda_VideoIO
     PRIVATE DO_SHAKTI_EXPORTS)
 else ()
   target_compile_definitions(DO_Shakti_Cuda_VideoIO
      PUBLIC DO_SHAKTI_STATIC)
endif ()

add_library(DO::Shakti::Cuda::VideoIO ALIAS DO_Shakti_Cuda_VideoIO)
