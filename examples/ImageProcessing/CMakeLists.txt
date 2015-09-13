cmake_minimum_required(VERSION 2.6)

project(DO_ImageProcessing_Examples)

message(STATUS "  - DO_ImageProcessing_Examples")

find_package(DO_Sara COMPONENTS Core Graphics ImageProcessing REQUIRED)

macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} ${DO_Sara_LIBRARIES})
  set_property(
    TARGET ${example} PROPERTY
    FOLDER "DO Sara Examples/ImageProcessing")
endmacro()

sara_add_example(image_processing_example)
