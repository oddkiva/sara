cmake_minimum_required(VERSION 2.6)

project(DO_Graphics_test)

message(STATUS "  - DO_Graphics_test")

find_package(DO COMPONENTS Core Graphics REQUIRED)
include(${DO_USE_FILE})

do_unit_test(Graphics_Qt test.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_Graphics_Qt_test PROPERTY FOLDER "DO Tests/Graphics")