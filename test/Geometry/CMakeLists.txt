cmake_minimum_required(VERSION 2.6)

project(DO_Geometry_tests)

message(STATUS "  - DO_Geometry_Tests")

find_package(DO COMPONENTS Core Graphics Geometry REQUIRED)
include(${DO_USE_FILE})

do_test(Geometry_BBox
        test_bbox.cpp
        "${DO_LIBRARIES}" "Geometry")

do_test(Geometry_SutherlandHodgman
        test_polygon_clipping_sutherland_hodgman.cpp 
        "${DO_LIBRARIES}" "Geometry")

do_test(Geometry_PolynomialRoots
        test_polynomial_roots.cpp
        "${DO_LIBRARIES}" "Geometry")