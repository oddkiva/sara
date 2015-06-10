cmake_minimum_required(VERSION 2.6)

project(DO_Features_tests)

message(STATUS "  - DO_Features_tests")

find_package(DO_Sara COMPONENTS Graphics Features ImageProcessing REQUIRED)

do_test(Features_DataStructures test_features.cpp "${DO_LIBRARIES}" "Features")
