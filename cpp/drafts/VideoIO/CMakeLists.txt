find_package(SaraFFMPEG REQUIRED)

add_library(DO_Sara_Audio AudioStream.hpp AudioStream.cpp)

target_include_directories(DO_Sara_AudioStream
  PRIVATE
  ${FFMPEG_INCLUDE_DIRS}
  PUBLIC
  ${DO_Sara_INCLUDE_DIR}
  ${DO_Sara_ThirdParty_DIR}
  ${DO_Sara_ThirdParty_DIR}/eigen)
target_link_libraries(DO_Sara_AudioIO PRIVATE DO_Sara_Core ${FFMPEG_LIBRARIES})
