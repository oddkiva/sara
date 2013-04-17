cmake_minimum_required(VERSION 2.6)

project(DO_Features_test)

message(STATUS "  - (DO_Features_test)")

find_package(DO COMPONENTS Graphics Features ImageProcessing REQUIRED)
include(${DO_USE_FILE})

do_unit_test(Features "${DO_LIBRARIES}")
