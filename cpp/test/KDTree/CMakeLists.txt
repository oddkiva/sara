cmake_minimum_required(VERSION 2.6)

project(DO_KDTree_Tests)

message(STATUS "  - DO_KDTree_Tests")

find_package(DO_Sara COMPONENTS KDTree REQUIRED)

file(GLOB test_kdtree_SOURCE_FILES FILES *.cpp)

foreach (file ${test_kdtree_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(${filename} ${file} "${DO_Sara_LIBRARIES}" "KDTree")
endforeach ()
