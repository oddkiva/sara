cmake_minimum_required(VERSION 2.6)

project(DO_ImageProcessing_tests)

message(STATUS "  - DO_ImageProcessing_tests")

find_package(DO_Sara COMPONENTS Core ImageProcessing REQUIRED)


file(GLOB test_imageprocessing_SOURCE_FILES FILES test_*.cpp)

foreach (file ${test_imageprocessing_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  do_add_test(${filename} ${file} "${DO_Sara_LIBRARIES}" "ImageProcessing")
endforeach ()
