cmake_minimum_required(VERSION 2.6)

project(DO_Geometry_tests)

message(STATUS "  - DO_Geometry_Tests")

find_package(DO_Sara COMPONENTS Core Geometry REQUIRED)

file(GLOB test_geometry_SOURCE_FILES FILES *.cpp)

foreach (file ${test_geometry_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  do_add_test(${filename} ${file} "${DO_Sara_LIBRARIES}" "Geometry")
endforeach ()
