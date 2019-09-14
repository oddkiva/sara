cmake_minimum_required(VERSION 2.6)

project(DO_FeatureMatching_tests)


message(STATUS "  - DO_Match_tests")

find_package(DO_Sara COMPONENTS Core Features Match REQUIRED)
find_package(HDF5 COMPONENTS CXX REQUIRED)

file(GLOB test_match_SOURCE_FILES FILES *.cpp)

foreach (file ${test_match_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(${filename} ${file} "${DO_Sara_LIBRARIES}" "Match")
endforeach ()
