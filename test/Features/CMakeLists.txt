cmake_minimum_required(VERSION 2.6)

project(DO_Features_tests)

message(STATUS "  - DO_Features_tests")

find_package(DO COMPONENTS Graphics Features ImageProcessing REQUIRED)
include(${DO_USE_FILE})

do_test(Features test.cpp "${DO_LIBRARIES}" "Features")
