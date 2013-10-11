cmake_minimum_required(VERSION 2.6)

project(DO_Core_test)

message(STATUS "  - DO_Core_test")

find_package(DO COMPONENTS Core REQUIRED)
include(${DO_USE_FILE})

do_unit_test(Core test.cpp "${DO_LIBRARIES}")