cmake_minimum_required(VERSION 2.6)

project(DO_ImageIO_Tests)

message(STATUS "  - DO_ImageIO_Tests")

find_package(DO_Sara COMPONENTS ImageIO REQUIRED)

file(GLOB test_imageio_SOURCE_FILES FILES *.cpp)

foreach (file ${test_imageio_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  do_test(${filename} ${file} "${DO_Sara_LIBRARIES}" "ImageIO")
endforeach ()
