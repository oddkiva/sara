cmake_minimum_required(VERSION 2.6)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif ()

project(DO_Features_Examples)

message(STATUS "  - DO_Features_Examples")

find_package(DO COMPONENTS Core Graphics Features REQUIRED)

macro (do_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${DO_DEFINITIONS})
  target_link_libraries(${example} ${DO_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER "DO Examples/Features")
endmacro ()

do_add_example(features_read_write_example)
