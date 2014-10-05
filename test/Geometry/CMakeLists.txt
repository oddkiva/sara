cmake_minimum_required(VERSION 2.6)

project(DO_Geometry_tests)

message(STATUS "  - DO_Geometry_Tests")

find_package(DO COMPONENTS Core Graphics Geometry REQUIRED)
include(${DO_USE_FILE})

do_test(Geometry_PotPourri test.cpp "${DO_LIBRARIES}" "Geometry")
do_test(Geometry_SutherlandHodgman test_SutherlandHodgman.cpp 
        "${DO_LIBRARIES}" "Geometry")
do_test(Geometry_Polynomial test_Polynomial.cpp "${DO_LIBRARIES}" "Geometry")
