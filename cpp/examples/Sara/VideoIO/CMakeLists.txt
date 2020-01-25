cmake_minimum_required(VERSION 2.6)

project(DO_VideoIO_Examples)

message(STATUS "  - DO_VideoIO_Examples")

find_package(DO_Sara COMPONENTS Core Graphics VideoIO REQUIRED)

macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} PRIVATE ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER "DO Sara Examples/VideoIO")
endmacro ()

sara_add_example(video_read_example)
# message(FATAL_ERROR "FFMPEG_LIBRARIES = ${FFMPEG_LIBRARIES}")
target_link_libraries(video_read_example PRIVATE ${FFMPEG_LIBRARIES})
