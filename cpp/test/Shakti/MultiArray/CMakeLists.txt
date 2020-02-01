cmake_minimum_required(VERSION 2.6)

find_package(DO_Sara COMPONENTS Core ImageProcessing REQUIRED)

file(GLOB test_multiarray_SOURCE_FILES FILES test_*.cu)

foreach (file ${test_multiarray_SOURCE_FILES})
  shakti_add_test(SOURCES ${file} DEPENDENCIES DO::Shakti::Utilities)
endforeach ()
