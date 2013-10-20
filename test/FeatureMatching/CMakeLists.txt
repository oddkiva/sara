cmake_minimum_required(VERSION 2.6)

project(DO_FeatureMatching_test)


message(STATUS "  - DO_FeatureMatching_test")

find_package(DO COMPONENTS Geometry FeatureDetectors Match FeatureMatching REQUIRED)
include(${DO_USE_FILE})

do_unit_test(FeatureMatching test.cpp "${DO_LIBRARIES}")