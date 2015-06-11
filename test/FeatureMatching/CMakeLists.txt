cmake_minimum_required(VERSION 2.6)

project(DO_FeatureMatching_tests)


message(STATUS "  - DO_FeatureMatching_tests")

find_package(
  DO COMPONENTS Geometry Features FeatureDescriptors FeatureDetectors Match 
  FeatureMatching REQUIRED)
include(${DO_USE_FILE})

do_test(FeatureMatching test.cpp "${DO_Sara_LIBRARIES}" "FeatureMatching")
