cmake_minimum_required(VERSION 2.6)

project(DO_FeatureDetectors_test)

message(STATUS "  - DO_FeatureDetectors_test")

find_package(DO COMPONENTS Core Graphics Features FeatureDetectors REQUIRED)
include(${DO_USE_FILE})

set(TestGroupName "FeatureDetectors")
do_unit_test(FeatureDetectors_LoG test_LoG.cpp "${DO_LIBRARIES}" ${TestGroupName})
do_unit_test(FeatureDetectors_DoG test_DoG.cpp "${DO_LIBRARIES}" ${TestGroupName})
do_unit_test(FeatureDetectors_Harris test_Harris.cpp "${DO_LIBRARIES}" ${TestGroupName})
do_unit_test(FeatureDetectors_Hessian test_Hessian.cpp "${DO_LIBRARIES}" ${TestGroupName})
# TODO: fix this.
#do_unit_test(FeatureDetectors_AffineAdaptation test_AffineAdaptation.cpp "${DO_LIBRARIES}" ${TestGroupName})