cmake_minimum_required(VERSION 2.6)

project(DO_FeatureDetectors_test)

message(STATUS "  - (DO_FeatureDetectors_test)")

find_package(DO COMPONENTS Graphics Features FeatureDetectors REQUIRED)
include(${DO_USE_FILE})

add_executable(DO_FeatureDetectors_test_Wrappers test_Wrappers.cpp)
do_use_modules(DO_FeatureDetectors_test_Wrappers "${DO_LIBRARIES}")
target_link_libraries(DO_FeatureDetectors_test_Wrappers gtest)

add_executable(DO_FeatureDetectors_test_SIFT test_SIFT.cpp)
do_use_modules(DO_FeatureDetectors_test_SIFT "${DO_LIBRARIES}")
target_link_libraries(DO_FeatureDetectors_test_SIFT gtest)

add_executable(DO_FeatureDetectors_test_Harris test_Harris.cpp)
do_use_modules(DO_FeatureDetectors_test_Harris "${DO_LIBRARIES}")
target_link_libraries(DO_FeatureDetectors_test_Harris gtest)

add_executable(DO_FeatureDetectors_test_Hessian test_Hessian.cpp)
do_use_modules(DO_FeatureDetectors_test_Hessian "${DO_LIBRARIES}")
target_link_libraries(DO_FeatureDetectors_test_Hessian gtest)

add_executable(DO_FeatureDetectors_test_AffineAdaptation test_AffineAdaptation.cpp)
do_use_modules(DO_FeatureDetectors_test_AffineAdaptation "${DO_LIBRARIES}")
target_link_libraries(DO_FeatureDetectors_test_AffineAdaptation gtest)