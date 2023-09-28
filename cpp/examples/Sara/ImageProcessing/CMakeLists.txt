find_package(DO_Sara
  COMPONENTS
  Core Graphics ImageIO VideoIO                           # Classic stuff
  DisjointSets FeatureDetectors Geometry ImageProcessing  # Algorithmic modules
  Visualization
  REQUIRED)

macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} PRIVATE ${DO_Sara_LIBRARIES})
  set_property(
    TARGET ${example} PROPERTY
    FOLDER "Examples/Sara/ImageProcessing")
endmacro()

sara_add_example(image_processing_example)
sara_add_example(infinite_image_example)
sara_add_example(gemm_based_convolution_example)

sara_add_example(curvature_example)
sara_add_example(fast_marching_example)
sara_add_example(level_set_2d_example)

sara_add_example(optical_flow_example)

sara_add_example(watershed_example)

sara_add_example(find_aruco_marker)
target_link_libraries(find_aruco_marker PRIVATE Boost::filesystem)
if (TBB_FOUND)
  target_link_libraries(find_aruco_marker PRIVATE TBB::tbb)
endif ()
