cmake_minimum_required(VERSION 2.6)

project(DO_ImageProcessing_tests)

message(STATUS "  - DO_ImageProcessing_tests")

find_package(DO COMPONENTS Core ImageProcessing REQUIRED)
include(${DO_USE_FILE})


do_test(ImageProcessing_Deriche test_deriche_filter.cpp
        "${DO_LIBRARIES}" "ImageProcessing")
        
do_test(ImageProcessing_Interpolation test_interpolation.cpp
        "${DO_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_LinearFiltering test_linear_filtering.cpp
        "${DO_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_Differential test_differential.cpp
        "${DO_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_Determinant test_determinant.cpp
        "${DO_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_LocalExtremum test_local_extremum.cpp
        "${DO_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_ImagePyramid test_image_pyramid.cpp
        "${DO_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_Norms test_norms.cpp
        "${DO_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_Orientation test_orientation.cpp
        "${DO_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_SecondMomentMatrix test_second_moment_matrix.cpp
        "${DO_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_Scaling test_scaling.cpp
        "${DO_LIBRARIES}" "ImageProcessing")