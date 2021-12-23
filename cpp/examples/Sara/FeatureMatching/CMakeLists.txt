find_package(DO_Sara COMPONENTS
  Core ImageIO VideoIO
  ImageProcessing
  Features FeatureDetectors FeatureDescriptors
  KDTree Match FeatureMatching SfM
  Visualization REQUIRED)

macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  target_compile_options(${example} PRIVATE ${SARA_DEFINITIONS})
  target_link_libraries(${example} PRIVATE ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example}
    PROPERTY
    FOLDER "Examples/Sara/FeatureMatching")
endmacro ()

sara_add_example(image_sift_matching)

sara_add_example(video_edge_matching)
target_link_libraries(video_edge_matching
  PRIVATE
  Boost::filesystem)

sara_add_example(video_sift_matching)
target_link_libraries(video_sift_matching
  PRIVATE
  Boost::filesystem
  Boost::program_options)
