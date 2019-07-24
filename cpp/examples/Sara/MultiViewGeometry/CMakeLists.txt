project(DO_MultiViewGeometry_Examples)

message(STATUS "  - DO_FeatureMatching_Examples")

find_package(DO_Sara COMPONENTS
  Core Graphics ImageIO Features FeatureDetectors FeatureDescriptors KDTree
  Match FeatureMatching MultiViewGeometry REQUIRED)


macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
    COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} PRIVATE ${DO_Sara_LIBRARIES})
  set_property(
    TARGET ${example} PROPERTY
    FOLDER "DO Sara Examples/ImageProcessing")
endmacro()


sara_add_example(homography_estimation_example)
sara_add_example(fundamental_matrix_estimation_example)
sara_add_example(essential_matrix_estimation_example.2)
target_link_libraries(essential_matrix_estimation_example.2 PRIVATE tinyply)
