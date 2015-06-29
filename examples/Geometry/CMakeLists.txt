cmake_minimum_required(VERSION 2.6)

project(DO_Geometry_Examples)

message(STATUS "  - DO_Geometry_Examples")

find_package(DO_Sara COMPONENTS Core Graphics Geometry REQUIRED)

macro (do_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${DO_DEFINITIONS})
  target_link_libraries(${example} ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER "DO Sara Examples/Geometry")
endmacro()

do_add_example(polygon_clipping_sutherland_hodgman_example)
