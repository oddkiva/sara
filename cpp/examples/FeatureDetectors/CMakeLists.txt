cmake_minimum_required(VERSION 2.6)

project(DO_FeatureDetectors_Examples)

message(STATUS "  - DO_FeatureDetectors_Examples")

find_package(DO_Sara COMPONENTS Graphics Features FeatureDetectors REQUIRED)
if (UNIX)
  find_package(HDF5 COMPONENTS CXX REQUIRED)
endif ()

macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example}
    PROPERTY
    FOLDER "DO Sara Examples/FeatureDetectors")
endmacro ()

file(GLOB example_SOURCE_FILES FILES *example.cpp)

foreach (file ${example_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_example(${filename})
endforeach ()

target_link_libraries(save_features_to_hdf5_example ${HDF5_LIBRARIES})
