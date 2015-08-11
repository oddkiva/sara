cmake_minimum_required(VERSION 2.6)

project(DO_FeatureDetectors_tests)

message(STATUS "  - DO_FeatureDetectors_tests")

find_package(DO_Sara COMPONENTS
            Graphics Features FeatureDetectors FeatureDescriptors REQUIRED)

file(GLOB test_feature_descriptors_SOURCE_FILES FILES *.cpp)
foreach (file ${test_feature_descriptors_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  do_add_test(${filename} ${file} "${DO_Sara_LIBRARIES}"
    "FeatureDescriptors")
endforeach ()

