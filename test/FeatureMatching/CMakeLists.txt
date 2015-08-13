cmake_minimum_required(VERSION 2.6)

project(DO_FeatureMatching_tests)


message(STATUS "  - DO_FeatureMatching_tests")

find_package(DO_Sara COMPONENTS
  Geometry Features FeatureDescriptors FeatureDetectors
  Match FeatureMatching REQUIRED)

do_add_test(test_feature_matching test_feature_matching.cpp
  "${DO_Sara_LIBRARIES}" "FeatureMatching")
