cmake_minimum_required(VERSION 2.6)

project(DO_ImageProcessing_test)

message(STATUS "  - DO_ImageProcessing_test")

find_package(DO COMPONENTS Core Graphics ImageProcessing REQUIRED)
include(${DO_USE_FILE})

set(TestGroupName "ImageProcessing")
do_unit_test(ImageProcessing_filter test.cpp "${DO_LIBRARIES}" ${TestGroupName})
do_unit_test(ImageProcessing_local_extrema test2.cpp "${DO_LIBRARIES}" ${TestGroupName})
do_unit_test(ImageProcessing_benchmark benchmark.cpp "${DO_LIBRARIES}" ${TestGroupName})