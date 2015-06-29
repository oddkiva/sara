cmake_minimum_required(VERSION 2.6)

project(DO_Features_Examples)

message(STATUS "  - DO_Features_Examples")

find_package(DO_Sara COMPONENTS Core Graphics Features REQUIRED)

macro (do_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${DO_DEFINITIONS})
  target_link_libraries(${example} ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER "DO Sara Examples/Features")
endmacro ()

do_add_example(features_read_write_example)
