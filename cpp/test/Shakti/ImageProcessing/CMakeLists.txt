cmake_minimum_required(VERSION 2.6)

project(DO_ImageProcessing_tests)

message(STATUS "  - DO_ImageProcessing_tests")

find_package(DO_Sara COMPONENTS Core ImageProcessing REQUIRED)


file(GLOB test_imageprocessing_SOURCE_FILES FILES test_*.cpp)

foreach (file ${test_imageprocessing_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  add_executable(${filename} ${file})
  target_link_libraries(${filename}
    gtest DO_Shakti_Utilities DO_Shakti_ImageProcessing)
  set_property(
    TARGET ${filename}
    PROPERTY
    FOLDER "DO Shakti Tests/ImageProcessing")
  set_target_properties(
    ${filename}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  add_test(${filename} ${CMAKE_BINARY_DIR}/bin/${filename})
endforeach ()
