cmake_minimum_required(VERSION 2.6)

project(DO_Match_test)

message(STATUS "  - DO_Match_tests")

find_package(DO_Sara COMPONENTS Core Graphics Match REQUIRED)
include(${DO_USE_FILE})

do_test(Match test.cpp "${DO_Sara_LIBRARIES}" "Match")
