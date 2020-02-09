if (NOT NV_VIDEOCODEC_ROOT)
  message(FATAL_ERROR "Please specify path to nVidia Video Codec SDK folder!")
  return ()
endif ()

find_package(NvidiaVideoCodec)

file(GLOB_RECURSE NV_VIDEOCODEC_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}
  *.cpp *.h *.cu)

cuda_add_library(nvidia_VideoCodecHelpers ${NV_VIDEOCODEC_SOURCE_FILES})
target_link_libraries(nvidia_VideoCodecHelpers nvidia::VideoCodec)
add_library(nvidia::VideoCodecHelpers ALIAS nvidia_VideoCodecHelpers)
