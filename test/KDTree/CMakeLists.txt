cmake_minimum_required(VERSION 2.6)

project(DO_KDTree_tests)

message(STATUS "  - DO_KDTree_tests")

find_package(DO COMPONENTS Graphics Geometry KDTree REQUIRED)
include(${DO_USE_FILE})

do_test(KDTree test.cpp "${DO_LIBRARIES}" "KDTree")
