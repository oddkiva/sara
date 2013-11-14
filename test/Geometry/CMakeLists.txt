cmake_minimum_required(VERSION 2.6)

project(DO_Geometry_test)

message(STATUS "  - DO_Geometry_test")

find_package(DO COMPONENTS Graphics Geometry REQUIRED)
include(${DO_USE_FILE})

do_unit_test(Geometry test.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_Geometry_test PROPERTY FOLDER "DO Tests/Geometry")