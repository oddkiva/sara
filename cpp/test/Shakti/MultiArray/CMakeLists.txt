cmake_minimum_required(VERSION 2.6)

project(DO_MultiArray_tests)

message(STATUS "  - DO_MultiArray_tests")

find_package(DO_Sara COMPONENTS Core ImageProcessing REQUIRED)


file(GLOB test_imageprocessing_SOURCE_FILES FILES test_*.cu)

foreach (file ${test_imageprocessing_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  cuda_add_executable(${filename} ${file})
  target_link_libraries(${filename}
    gtest
    DO_Shakti_Utilities)
  set_property(
    TARGET ${filename}
    PROPERTY
    FOLDER "DO Shakti Tests/MultiArray")
  set_target_properties(
    ${filename}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  add_test(${filename} ${CMAKE_BINARY_DIR}/bin/${filename})
endforeach ()
