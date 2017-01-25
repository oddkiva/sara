cmake_minimum_required(VERSION 2.6)

project(DO_VideoIO_Tests)

message(STATUS "  - DO_VideoIO_Tests")

find_package(DO_Sara COMPONENTS Core VideoIO REQUIRED)

file(GLOB test_videoio_SOURCE_FILES FILES *.cpp)

foreach (file ${test_videoio_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(${filename} ${file} "${DO_Sara_LIBRARIES}" "VideoIO")
endforeach ()
