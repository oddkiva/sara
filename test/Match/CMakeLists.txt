cmake_minimum_required(VERSION 2.6)

project(DO_FeatureMatching_tests)


message(STATUS "  - DO_Match_tests")

find_package(DO_Sara COMPONENTS Match REQUIRED)

do_add_test(test_match test_match.cpp "${DO_Sara_LIBRARIES}" "Match")
