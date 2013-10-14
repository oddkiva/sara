cmake_minimum_required(VERSION 2.6)

project(DO_Graphics_test)

message(STATUS "  - DO_Graphics_test")

find_package(DO COMPONENTS Core Graphics REQUIRED)
include(${DO_USE_FILE})

do_unit_test(Graphics_Qt test.cpp "${DO_LIBRARIES}")

add_library(ImageIO ImageIO.hpp ImageIO.cpp)
target_link_libraries(ImageIO libjpeg libpng libtiff)
do_unit_test(Graphics_ImageIO test_ImageIO.cpp "${DO_LIBRARIES}")
target_link_libraries(DO_Graphics_ImageIO_test ImageIO)