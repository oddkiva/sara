cmake_minimum_required(VERSION 2.6)

project(DO_ImageDrawing_tests)

message(STATUS "  - DO_ImageDrawing_tests")

find_package(DO COMPONENTS Core ImageDrawing Graphics FileSystem REQUIRED)
include(${DO_USE_FILE})

do_test(ImageDrawing_ImageIO test_ImageIO.cpp "${DO_LIBRARIES}" "ImageDrawing")
