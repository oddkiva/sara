cmake_minimum_required(VERSION 2.6)

project(DO_KDTree_Tests)

message(STATUS "  - DO_KDTree_Tests")

find_package(DO_Sara COMPONENTS KDTree REQUIRED)

do_test(KDTree_KNNSearch
        test_knn_search.cpp
        "${DO_LIBRARIES}" "KDTree")

do_test(KDTree_RadiusSearch
        test_radius_search.cpp
        "${DO_LIBRARIES}" "KDTree")
