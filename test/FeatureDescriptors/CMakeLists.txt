cmake_minimum_required(VERSION 2.6)

project(DO_FeatureDetectors_tests)

message(STATUS "  - DO_FeatureDetectors_tests")

find_package(DO_Sara COMPONENTS
            Graphics Features FeatureDetectors FeatureDescriptors REQUIRED)
include(${DO_USE_FILE})

do_test(FeatureDescriptors_SIFT test_SIFT.cpp "${DO_LIBRARIES}"
        "FeatureDescriptors")
#do_unit_test(FeatureDescriptors_DAISY test_DAISY.cpp "${DO_LIBRARIES}")
