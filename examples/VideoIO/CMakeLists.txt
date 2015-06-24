cmake_minimum_required(VERSION 2.6)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif ()

project(DO_VideoIO_Examples)

message(STATUS "  - DO_VideoIO_Examples")

find_package(DO_Sara COMPONENTS Core Graphics VideoIO REQUIRED)

macro (do_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${DO_DEFINITIONS})
  target_link_libraries(${example} ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER "DO Sara Examples/VideoIO")
endmacro ()

do_add_example(video_read_example)
