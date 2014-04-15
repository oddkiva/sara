cmake_minimum_required(VERSION 2.6)

project(DO_FeatureDetectors_test)

message(STATUS "  - DO_FeatureDetectors_test")

find_package(DO COMPONENTS Core Graphics Features FeatureDetectors 
  FeatureDescriptors REQUIRED)
include(${DO_USE_FILE})

set(TestGroupName "FeatureDescriptors")
do_unit_test(FeatureDescriptors_SIFT test_SIFT.cpp "${DO_LIBRARIES}" ${TestGroupName})
#do_unit_test(FeatureDescriptors_DAISY test_DAISY.cpp "${DO_LIBRARIES}")