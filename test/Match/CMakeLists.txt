cmake_minimum_required(VERSION 2.6)

project(DO_Match_test)

message(STATUS "  - DO_Match_test")

find_package(DO COMPONENTS Core Graphics Match REQUIRED)
include(${DO_USE_FILE})

set(TestGroupName "Match")
do_unit_test(Match test.cpp "${DO_LIBRARIES}" ${TestGroupName})