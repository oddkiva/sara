project(DO_MultiViewGeometry_Examples)

message(STATUS "  - DO_MultiViewGeometry_Examples")

find_package(DO_Sara COMPONENTS
  Core Graphics ImageIO Features FeatureDetectors FeatureDescriptors KDTree
  Match FeatureMatching MultiViewGeometry SfM REQUIRED)


macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
    COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} PRIVATE ${DO_Sara_LIBRARIES})
  set_property(
    TARGET ${example} PROPERTY
    FOLDER "DO Sara Examples/MultiViewGeometry")
endmacro()


sara_add_example(homography_estimation_example)
sara_add_example(fundamental_matrix_estimation_example)
sara_add_example(essential_matrix_estimation_example)
target_link_libraries(essential_matrix_estimation_example PRIVATE tinyply)
