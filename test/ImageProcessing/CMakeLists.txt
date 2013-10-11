cmake_minimum_required(VERSION 2.6)

project(DO_ImageProcessing_test)

message(STATUS "  - DO_ImageProcessing_test")

find_package(DO COMPONENTS Core Graphics ImageProcessing REQUIRED)
include(${DO_USE_FILE})

do_unit_test(ImageProcessing_1 test2.cpp "${DO_LIBRARIES}")
do_unit_test(DO_ImageProcessing_2 test2.cpp "${DO_LIBRARIES}")
do_unit_test(DO_ImageProcessing_benchmark benchmark.cpp "${DO_LIBRARIES}")