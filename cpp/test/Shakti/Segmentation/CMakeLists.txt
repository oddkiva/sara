cmake_minimum_required(VERSION 2.6)

project(DO_Segmentation_tests)

message(STATUS "  - DO_Segmentation_tests")

find_package(DO_Sara COMPONENTS Core ImageProcessing REQUIRED)


file(GLOB test_imageprocessing_SOURCE_FILES FILES test_*.cu)

foreach (file ${test_imageprocessing_SOURCE_FILES})
  shakti_add_test(
    SOURCES ${file}
    DEPENDENCIES DO::Shakti::Utilities DO::Shakti::Segmentation)
endforeach ()
