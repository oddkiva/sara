project(DO_MultiViewGeometry_Examples)

message(STATUS "  - DO_FeatureMatching_Examples")

find_package(DO_Sara COMPONENTS
  Core Graphics ImageIO Features FeatureDetectors FeatureDescriptors KDTree
  Match FeatureMatching MultiViewGeometry REQUIRED)

add_executable(essential_matrix_estimation_example
  essential_matrix_estimation_example.cpp)

set_target_properties(essential_matrix_estimation_example PROPERTIES
  COMPILE_FLAGS ${SARA_DEFINITIONS})
target_link_libraries(essential_matrix_estimation_example ${DO_Sara_LIBRARIES}
  tinyply)

set_property(TARGET essential_matrix_estimation_example
  PROPERTY
  FOLDER "DO Sara Examples/MultiViewGeometry")
