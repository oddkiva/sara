find_package(DO_Sara COMPONENTS FeatureDetectors ImageIO ImageProcessing VideoIO Visualization REQUIRED)

macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} PRIVATE ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example}
    PROPERTY
    FOLDER "Examples/Sara/FeatureDetectors")
endmacro ()

file(GLOB example_SOURCE_FILES FILES *example.cpp)

foreach (file ${example_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_example(${filename})
endforeach ()

target_link_libraries(edge_detection_example
  PRIVATE
  Boost::filesystem
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
target_compile_definitions(edge_detection_example
  PRIVATE
  -DBOOST_ALL_DYN_LINK
  -DBOOST_ALL_NO_LIB)

target_link_libraries(line_segment_detection_example
  PRIVATE $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
