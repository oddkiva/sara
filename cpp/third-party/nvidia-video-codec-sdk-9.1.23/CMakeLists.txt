if (NOT NvidiaVideoCodec_ROOT)
  message(FATAL_ERROR "Please specify path to nVidia Video Codec SDK folder!")
  return ()
endif ()

find_package(NvidiaVideoCodec)

set(NvDecoder_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvDecoder/NvDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvDecoder/NvDecoder.h)

list (APPEND NvEncoder_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvEncoder/NvEncoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvEncoder/NvEncoder.h
  ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvEncoder/NvEncoderCuda.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvEncoder/NvEncoderCuda.h
  ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvEncoder/NvEncoderGL.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvEncoder/NvEncoderGL.h
  ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvEncoder/NvEncoderOutputInVidMemCuda.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvEncoder/NvEncoderOutputInVidMemCuda.h)

if (WIN32 AND USE_DIRECT3D)
  list (APPEND NvEncoder_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvEncoder/NvEncoderD3D11.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvEncoder/NvEncoderD3D11.h
    ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvEncoder/NvEncoderD3D9.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvEncoder/NvEncoderD3D9.h)
endif ()

file (GLOB Utils_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils/*.cu)

cuda_add_library(nvidia_VideoCodecHelpers
  ${NvDecoder_SOURCE_FILES}
  ${NvEncoder_SOURCE_FILES}
  ${Utils_SOURCE_FILES})
target_include_directories(nvidia_VideoCodecHelpers
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/NvCodec>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/NvDecoder>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/NvEncoder>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Utils>)
target_link_libraries(nvidia_VideoCodecHelpers nvidia::VideoCodec)
target_compile_options(nvidia_VideoCodecHelpers
  PRIVATE
  -Wno-missing-field-initializers
  -Wno-reorder
  -Wno-sign-compare
  -Wno-switch
  -Wno-unused-parameter
  -Wno-unused-variable)
add_library(nvidia::VideoCodecHelpers ALIAS nvidia_VideoCodecHelpers)
