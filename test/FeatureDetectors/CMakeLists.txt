cmake_minimum_required(VERSION 2.6)

project(DO_FeatureDetectors_test)

message(STATUS "  - DO_FeatureDetectors_test")

find_package(DO COMPONENTS Graphics Features FeatureDetectors REQUIRED)
include(${DO_USE_FILE})

do_unit_test(FeatureDetectors_LoG test_LoG.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_FeatureDetectors_LoG_test PROPERTY FOLDER "DO Tests/FeatureDetectors")

do_unit_test(FeatureDetectors_DoG test_DoG.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_FeatureDetectors_DoG_test PROPERTY FOLDER "DO Tests/FeatureDetectors")

do_unit_test(FeatureDetectors_Harris test_Harris.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_FeatureDetectors_Harris_test PROPERTY FOLDER "DO Tests/FeatureDetectors")

do_unit_test(FeatureDetectors_Hessian test_Hessian.cpp "${DO_LIBRARIES}")
set_property(TARGET DO_FeatureDetectors_Hessian_test PROPERTY FOLDER "DO Tests/FeatureDetectors")
# TODO: fix this.
#do_unit_test(FeatureDetectors_AffineAdaptation test_AffineAdaptation.cpp "${DO_LIBRARIES}")