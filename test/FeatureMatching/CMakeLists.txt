cmake_minimum_required(VERSION 2.6)

project(DO_FeatureMatching_tests)


message(STATUS "  - DO_FeatureMatching_tests")

find_package(DO_Sara COMPONENTS
  Geometry Features FeatureDescriptors FeatureDetectors
  Match FeatureMatching REQUIRED)

file(GLOB test_featurematching_SOURCE_FILES FILES *.cpp)

foreach (file ${test_featurematching_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  do_add_test(${filename} ${file} "${DO_Sara_LIBRARIES}" "FeatureMatching")
endforeach ()
