cmake_minimum_required(VERSION 2.6)

project(DO_ImageIO_Tests)

message(STATUS "  - DO_ImageIO_Tests")

find_package(DO_Sara COMPONENTS ImageIO REQUIRED)

do_test(ImageIO_IO test_image_io.cpp "${DO_Sara_LIBRARIES}" "ImageIO")
