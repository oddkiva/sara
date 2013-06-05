cmake_minimum_required(VERSION 2.6)

project(DO_ImageProcessing_test)

message(STATUS "  - DO_ImageProcessing_test")

find_package(DO COMPONENTS Core Graphics ImageProcessing REQUIRED)
include(${DO_USE_FILE})

do_unit_test(ImageProcessing "${DO_LIBRARIES}")

add_executable(DO_ImageProcessing_test2 test2.cpp)
do_use_modules(DO_ImageProcessing_test2 "${DO_LIBRARIES}")
target_link_libraries(DO_ImageProcessing_test2 gtest)

add_executable(DO_ImageProcessing_benchmark benchmark.cpp)
do_use_modules(DO_ImageProcessing_benchmark "${DO_LIBRARIES}")
target_link_libraries(DO_ImageProcessing_benchmark gtest)