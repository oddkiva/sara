cmake_minimum_required(VERSION 2.6)

project(DO_Graphics_test)

message(STATUS "  - DO_Graphics_test")


find_package(DO COMPONENTS Core Graphics REQUIRED)
include(${DO_USE_FILE})

set(SRCS GraphicsTesting.hpp test.cpp)
do_add_msvc_precompiled_header("GraphicsTesting.hpp" SRCS)
do_unit_test(Graphics_Qt "${SRCS}" "${DO_LIBRARIES}")
#set_target_properties(DO_Graphics_Qt_test PROPERTIES
#                      COTIRE_CXX_PREFIX_HEADER_INIT "GraphicsTesting.hpp"
#                      COTIRE_UNITY_LINK_LIBRARIES_INIT "COPY")
#cotire(DO_Graphics_Qt_test)
#target_link_libraries(DO_Graphics_Qt_test_unity "gtest;${DO_LIBRARIES}")
set_property(TARGET DO_Graphics_Qt_test PROPERTY FOLDER "DO Tests/Graphics")
