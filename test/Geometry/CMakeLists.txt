cmake_minimum_required(VERSION 2.6)

project(DO_Geometry_test)

message(STATUS "  - DO_Geometry_test")

find_package(DO COMPONENTS Graphics Geometry REQUIRED)
include(${DO_USE_FILE})

do_unit_test(Geometry_PotPourri test.cpp "${DO_LIBRARIES}" "Geometry")
do_unit_test(Geometry_SutherlandHodgman test_SutherlandHodgman.cpp 
             "${DO_LIBRARIES}" "Geometry")
do_unit_test(Geometry_Polynomial test_Polynomial.cpp
             "${DO_LIBRARIES}" "Geometry")
