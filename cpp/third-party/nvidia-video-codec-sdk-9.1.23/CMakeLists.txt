find_package(NvidiaVideoCodec)
find_package(GLEW REQUIRED)

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

cuda_add_library(nvidia_VideoCodecHelpers STATIC
  ${NvDecoder_SOURCE_FILES}
  ${NvEncoder_SOURCE_FILES}
  ${Utils_SOURCE_FILES})
target_include_directories(nvidia_VideoCodecHelpers
  PUBLIC
  ${GLEW_INCLUDE_DIRS}
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/NvCodec>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/NvDecoder>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/NvEncoder>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Utils>)
target_link_libraries(nvidia_VideoCodecHelpers
  PUBLIC
  nvidia::VideoCodec
  ${GLEW_LIBRARIES})
target_compile_options(nvidia_VideoCodecHelpers
  PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-missing-field-initializers>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-reorder>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-switch>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-parameter>
  $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-variable>)
set_target_properties(nvidia_VideoCodecHelpers
  PROPERTIES
  FOLDER "Third-Party")
add_library(nvidia::VideoCodecHelpers ALIAS nvidia_VideoCodecHelpers)
