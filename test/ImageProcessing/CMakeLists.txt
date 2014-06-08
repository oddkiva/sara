cmake_minimum_required(VERSION 2.6)

project(DO_ImageProcessing_tests)

message(STATUS "  - DO_ImageProcessing_tests")

find_package(DO COMPONENTS Core Graphics ImageProcessing REQUIRED)
include(${DO_USE_FILE})

# add_library(DO_AKAZE
            # FED.hpp FED.cpp
            # NonlinearDiffusion.hpp NonlinearDiffusion.cpp)

do_test(ImageProcessing_filter test.cpp "${DO_LIBRARIES}" "ImageProcessing")
do_test(ImageProcessing_local_extrema test2.cpp "${DO_LIBRARIES}"
        "ImageProcessing")
do_test(ImageProcessing_benchmark benchmark.cpp "${DO_LIBRARIES}"
        "ImageProcessing")
