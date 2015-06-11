cmake_minimum_required(VERSION 2.6)

project(DO_Geometry_tests)

message(STATUS "  - DO_Geometry_Tests")

message("DO_Sara_ImageProcessing_LIBRARIES = ${DO_Sara_ImageProcessing_LIBRARIES}")

find_package(DO_Sara COMPONENTS Core Geometry REQUIRED)

do_test(Geometry_BBox
        test_bbox.cpp
        "${DO_Sara_LIBRARIES}" "Geometry")

do_test(Geometry_Cone
        test_cone.cpp
        "${DO_Sara_LIBRARIES}" "Geometry")

do_test(Geometry_CSG
        test_csg.cpp
        "${DO_Sara_LIBRARIES}" "Geometry")

do_test(Geometry_Ellipse
        test_ellipse.cpp
        "${DO_Sara_LIBRARIES}" "Geometry")

do_test(Geometry_Quad
        test_quad.cpp
        "${DO_Sara_LIBRARIES}" "Geometry")

do_test(Geometry_Triangle
        test_triangle.cpp
        "${DO_Sara_LIBRARIES}" "Geometry")

do_test(Geometry_SutherlandHodgman
        test_polygon_clipping_sutherland_hodgman.cpp 
        "${DO_Sara_LIBRARIES}" "Geometry")

do_test(Geometry_PolynomialRoots
        test_polynomial_roots.cpp
        "${DO_Sara_LIBRARIES}" "Geometry")
