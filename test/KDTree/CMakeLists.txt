cmake_minimum_required(VERSION 2.6)

project(DO_KDTree_test)

message(STATUS "  - DO_KDTree_test")

find_package(DO COMPONENTS Graphics Geometry KDTree REQUIRED)
include(${DO_USE_FILE})

do_unit_test(KDTree test.cpp "${DO_LIBRARIES}")