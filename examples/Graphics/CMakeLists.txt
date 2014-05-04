cmake_minimum_required(VERSION 2.6)

project(DO_Graphics_example)

message(STATUS "  - DO_Graphics_example")

find_package(DO COMPONENTS Core Graphics REQUIRED)
include(${DO_USE_FILE})

# TODO: make precompiled support easier. This is not maintainable.
# This should be along these lines:
#set(SRCS example.cpp GraphicsExample.hpp)
#do_add_msvc_precompiled_header("GraphicsExample.hpp" SRCS "GraphicsExample_PCH")

add_executable(DO_Graphics_example example.cpp precompiled.hpp)
# TODO: if C++11 is available, activate it **GLOBALLY**.
# TODO: improve this, this is not very user-friendly!
set_target_properties(DO_Graphics_example PROPERTIES
                      COMPILE_FLAGS "${ENABLE_CXX11} -DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}"
                      COMPILE_DEFINITIONS DO_STATIC)
target_link_libraries(DO_Graphics_example ${DO_LIBRARIES})

# Group this project as part of the set of sample programs.
set_property(TARGET DO_Graphics_example PROPERTY FOLDER "DO Examples")
