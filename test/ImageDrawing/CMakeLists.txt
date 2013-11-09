cmake_minimum_required(VERSION 2.6)

project(DO_ImageDrawing_test)

message(STATUS "  - DO_ImageDrawing_test")

find_package(DO COMPONENTS Core REQUIRED)
include(${DO_USE_FILE})

#do_unit_test(ImageDrawing_ImageIO test_ImageIO.cpp "${DO_LIBRARIES}")
include_directories(
  ${antigrain_DIR}
  ${jpeg_DIR}
  ${png_DIR}
  ${tiff_DIF} )

add_executable(DO_ImageDrawing_ImageIO_test
               test_ImageIO.cpp ImageIO.hpp ImageIO.cpp)
do_use_modules(DO_ImageDrawing_ImageIO_test "${DO_LIBRARIES};jpeg;antigrain;png;tiff")