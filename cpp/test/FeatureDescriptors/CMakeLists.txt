cmake_minimum_required(VERSION 2.6)

project(DO_FeatureDescriptors_tests)

message(STATUS "  - DO_FeatureDescriptors_tests")

find_package(DO_Sara COMPONENTS
  Features FeatureDetectors FeatureDescriptors
  Graphics ImageIO
  REQUIRED)

file(GLOB test_feature_descriptors_SOURCE_FILES FILES *.cpp)
foreach (file ${test_feature_descriptors_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(${filename} ${file} "${DO_Sara_LIBRARIES}"
    "FeatureDescriptors")
endforeach ()

