cmake_minimum_required(VERSION 2.6)

project(DO_ImageIO_Tests)

message(STATUS "  - DO_ImageIO_Tests")

find_package(DO COMPONENTS ImageIO REQUIRED)
include(${DO_USE_FILE})

do_test(ImageIO_IO test_image_io.cpp "${DO_LIBRARIES}" "ImageIO")
