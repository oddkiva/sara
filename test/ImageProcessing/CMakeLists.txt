cmake_minimum_required(VERSION 2.6)

project(DO_ImageProcessing_tests)

message(STATUS "  - DO_ImageProcessing_tests")

find_package(DO COMPONENTS Core ImageProcessing REQUIRED)
include(${DO_USE_FILE})


do_test(ImageProcessing_Deriche test_deriche_filter.cpp
        "${DO_LIBRARIES}" "ImageProcessing")
        
do_test(ImageProcessing_Interpolation test_interpolation.cpp
        "${DO_LIBRARIES}" "ImageProcessing")