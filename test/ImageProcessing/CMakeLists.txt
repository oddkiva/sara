cmake_minimum_required(VERSION 2.6)

project(DO_ImageProcessing_test)

message(STATUS "  - DO_ImageProcessing_test")

find_package(DO COMPONENTS Core Graphics ImageProcessing REQUIRED)
include(${DO_USE_FILE})

do_unit_test(ImageProcessing "${DO_LIBRARIES}")