cmake_minimum_required(VERSION 2.6)

project(DO_ImageDrawing_test)

message(STATUS "  - DO_ImageDrawing_test")

find_package(DO COMPONENTS Core ImageDrawing Graphics REQUIRED)
include(${DO_USE_FILE})

do_unit_test(ImageDrawing_ImageIO test_ImageIO.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_ImageDrawing_ImageIO_test PROPERTY FOLDER "DO Tests/ImageDrawing")