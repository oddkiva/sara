find_package(DO_Sara COMPONENTS
  Core Graphics ImageIO VideoIO
  ImageProcessing
  Features FeatureDetectors FeatureDescriptors
  KDTree Match FeatureMatching
  MultiViewGeometry SfM Visualization REQUIRED)

macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
    COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} PRIVATE ${DO_Sara_LIBRARIES})
  set_property(
    TARGET ${example} PROPERTY
    FOLDER "Examples/Sara/MultiViewGeometry")
endmacro()


# Camera model.
sara_add_example(fisheye_camera_example)
sara_add_example(metrology_example)
sara_add_example(omnidirectional_camera_example)

# Single view geometry.
sara_add_example(metric_birds_eye_viewer)
sara_add_example(orthogonal_vanishing_point_detection)
target_include_directories(orthogonal_vanishing_point_detection
  PRIVATE ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(orthogonal_vanishing_point_detection
  PRIVATE
  Boost::filesystem
  Boost::program_options
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)

# Two-view geometry.
sara_add_example(homography_estimation_example)
sara_add_example(fundamental_7_point_example)
sara_add_example(essential_5_point_example)
sara_add_example(relative_pose_estimation_example)
target_link_libraries(essential_5_point_example PRIVATE tinyply)

# Camera resectioning.
sara_add_example(camera_resectioning_example)

# Visual odometry.
sara_add_example(visual_odometry_example)

# Bundle adjustment.
sara_add_example(two_view_bundle_adjustment_example)
target_include_directories(two_view_bundle_adjustment_example
  PRIVATE
  ${CERES_INCLUDE_DIRS})
target_link_libraries(two_view_bundle_adjustment_example
  PRIVATE
  ${CERES_LIBRARIES})
