project(gtest CXX C)
cmake_minimum_required(VERSION 2.6.2)


include_directories(fused-src)

add_library(gtest fused-src/gtest/gtest-all.cc)
add_library(gtest_main fused-src/gtest/gtest_main.cc)
target_link_libraries(gtest_main gtest)