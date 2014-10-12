cmake_minimum_required(VERSION 2.6)

project(DO_KDTree_Tests)

message(STATUS "  - DO_KDTree_Tests")

find_package(DO COMPONENTS KDTree REQUIRED)
include(${DO_USE_FILE})

do_test(KDTree_KDTree test.cpp "${DO_LIBRARIES}" "KDTree")
