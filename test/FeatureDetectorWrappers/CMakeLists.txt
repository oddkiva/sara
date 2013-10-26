cmake_minimum_required(VERSION 2.6)

project(DO_FeatureDetectorWrappers_test)

message(STATUS "  - DO_FeatureDetectorWrappers_test")

find_package(DO COMPONENTS Graphics Features FeatureDetectorWrappers REQUIRED)
include(${DO_USE_FILE})

do_unit_test(FeatureDetectorWrappers test_Wrappers.cpp "${DO_LIBRARIES}")