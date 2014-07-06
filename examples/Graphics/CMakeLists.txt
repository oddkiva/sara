cmake_minimum_required(VERSION 2.6)

project(DO_Graphics_example)

message(STATUS "  - DO_Graphics_example")

find_package(DO COMPONENTS Core Graphics REQUIRED)
include(${DO_USE_FILE})

# TODO: make precompiled support easier. This is not maintainable.
# This should be along these lines:
#set(SRCS example.cpp GraphicsExample.hpp)
#do_add_msvc_precompiled_header("GraphicsExample.hpp" SRCS "GraphicsExample_PCH")

macro (do_add_executable _project_name _project_sources)
  add_executable(${_project_name} ${_project_sources})
  # TODO: if C++11 is available, activate it **GLOBALLY**.
  # TODO: improve this, this is not very user-friendly!
  set_target_properties(${_project_name} PROPERTIES
                        COMPILE_FLAGS "${ENABLE_CXX11} -DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}"
                        COMPILE_DEFINITIONS DO_STATIC)
  target_link_libraries(${_project_name} ${DO_LIBRARIES})
  # Group this project as part of the set of sample programs.
  set_property(TARGET ${_project_name} PROPERTY FOLDER "DO Examples/Graphics")
endmacro()

do_add_executable(DO_Graphics_example "example.cpp;precompiled.hpp")

do_add_executable(example_1 example_1.cpp)
do_add_executable(example_2 example_2.cpp)
do_add_executable(example_3 example_3.cpp)
do_add_executable(example_4 example_4.cpp)
do_add_executable(example_5 example_5.cpp)
do_add_executable(example_6 example_6.cpp)
do_add_executable(example_7 example_7.cpp)
do_add_executable(example_8 example_8.cpp)