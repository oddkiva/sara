cmake_minimum_required(VERSION 2.6)

project(DO_ImageProcessing_test)

message(STATUS "  - DO_ImageProcessing_test")

find_package(DO COMPONENTS Core Graphics ImageProcessing REQUIRED)
include(${DO_USE_FILE})

# add_library(DO_AKAZE
            # FED.hpp FED.cpp
            # NonlinearDiffusion.hpp NonlinearDiffusion.cpp)

do_unit_test(ImageProcessing_filter test.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_ImageProcessing_filter_test PROPERTY FOLDER "DO Tests/ImageProcessing")
do_unit_test(ImageProcessing_local_extrema test2.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_ImageProcessing_local_extrema_test PROPERTY FOLDER "DO Tests/ImageProcessing")
do_unit_test(ImageProcessing_benchmark benchmark.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_ImageProcessing_benchmark_test PROPERTY FOLDER "DO Tests/ImageProcessing")