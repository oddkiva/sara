find_package(DO_Sara COMPONENTS
  Core Graphics ImageIO VideoIO
  Features FeatureDetectors FeatureDescriptors
  KDTree Match FeatureMatching SfM REQUIRED)

macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  target_compile_options(${example} PRIVATE ${SARA_DEFINITIONS})
  target_link_libraries(${example} PRIVATE ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example}
    PROPERTY
    FOLDER "Examples/Sara/FeatureMatching")
endmacro ()

sara_add_example(sift_matching_example)

sara_add_example(edge_matching_example)
target_link_libraries(edge_matching_example PRIVATE Boost::filesystem)
