cmake_minimum_required(VERSION 2.6)

project(DO_ImageProcessing_tests)

message(STATUS "  - DO_ImageProcessing_tests")

find_package(DO_Sara COMPONENTS Core ImageProcessing REQUIRED)


do_test(ImageProcessing_Deriche test_deriche_filter.cpp
        "${DO_Sara_LIBRARIES}" "ImageProcessing")
        
do_test(ImageProcessing_Interpolation test_interpolation.cpp
        "${DO_Sara_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_LinearFiltering test_linear_filtering.cpp
        "${DO_Sara_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_Differential test_differential.cpp
        "${DO_Sara_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_Determinant test_determinant.cpp
        "${DO_Sara_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_LocalExtremum test_local_extremum.cpp
        "${DO_Sara_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_ImagePyramid test_image_pyramid.cpp
        "${DO_Sara_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_Norms test_norms.cpp
        "${DO_Sara_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_Orientation test_orientation.cpp
        "${DO_Sara_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_SecondMomentMatrix test_second_moment_matrix.cpp
        "${DO_Sara_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_Scaling test_scaling.cpp
        "${DO_Sara_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_GaussianPyramid test_gaussian_pyramid.cpp
        "${DO_Sara_LIBRARIES}" "ImageProcessing")

do_test(ImageProcessing_Warp test_image_warp.cpp
        "${DO_Sara_LIBRARIES}" "ImageProcessing")
