cmake_minimum_required(VERSION 2.6)

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

project(DO_Graphics_example)

message(STATUS "  - DO_Graphics_example")

find_package(DO COMPONENTS Core Graphics REQUIRED)
include(${DO_USE_FILE})

# TODO: make precompiled support easier. This is not maintainable.
# This should be along these lines:
#set(SRCS example.cpp GraphicsExample.hpp)
#do_add_msvc_precompiled_header("GraphicsExample.hpp"
#                               SRCS "GraphicsExample_PCH")

macro (do_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}"
                        COMPILE_DEFINITIONS DO_STATIC)
  target_link_libraries(${example} ${DO_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER "DO Examples/Graphics")
endmacro()

# Window management examples.
foreach(i RANGE 1 4)
  do_add_example(window_management_example_${i})
endforeach()
# Drawing examples.
foreach(i RANGE 1 2)
  do_add_example(drawing_example_${i})
endforeach()
# Image display examples.
foreach(i RANGE 1 3)
  do_add_example(image_example_${i})
endforeach()
# Keyboard and mouse input examples.
foreach(i RANGE 1 3)
  do_add_example(keyboard_mouse_event_example_${i})
endforeach()
# Dialog box example.
do_add_example(dialog_box_example)
# Mesh viewer example.
do_add_example(mesh_viewer_example)
# Interactive viewer example.
do_add_example(interactive_viewer_example)