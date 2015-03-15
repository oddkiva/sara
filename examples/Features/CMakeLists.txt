cmake_minimum_required(VERSION 2.6)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif ()

project(DO_Features_Examples)

message(STATUS "  - DO_Features_Examples")

find_package(DO COMPONENTS Core Graphics Features REQUIRED)
include(${DO_USE_FILE})

macro (do_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}"
                        COMPILE_DEFINITIONS DO_STATIC)
  target_link_libraries(${example} ${DO_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER "DO Examples/Features")
endmacro ()

do_add_example(features_read_write_example)
