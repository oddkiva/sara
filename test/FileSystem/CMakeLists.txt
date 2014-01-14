cmake_minimum_required(VERSION 2.6)

project(DO_FileSystem_test)

message(STATUS "  - DO_FileSystem_test")

find_package(DO COMPONENTS FileSystem REQUIRED)

include(${DO_USE_FILE})
do_unit_test(FileSystem test.cpp "${DO_LIBRARIES}" "FileSystem")