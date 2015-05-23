cmake_minimum_required(VERSION 2.6)

project(DO_FeatureDetectors_tests)

message(STATUS "  - DO_FeatureDetectors_tests")

find_package(DO COMPONENTS Graphics Features FeatureDetectors REQUIRED)
include(${DO_USE_FILE})

set(TEST_GROUP_NAME "FeatureDetectors")
do_test(FeatureDetectors_LoG test_LoG.cpp "${DO_LIBRARIES}" ${TEST_GROUP_NAME})
do_test(FeatureDetectors_DoG test_DoG.cpp "${DO_LIBRARIES}" ${TEST_GROUP_NAME})
do_test(FeatureDetectors_Harris test_Harris.cpp "${DO_LIBRARIES}"
        ${TEST_GROUP_NAME})
do_test(FeatureDetectors_Hessian test_Hessian.cpp "${DO_LIBRARIES}"
        ${TEST_GROUP_NAME})
#do_unit_test(FeatureDetectors_AffineAdaptation test_AffineAdaptation.cpp "${DO_LIBRARIES}")
