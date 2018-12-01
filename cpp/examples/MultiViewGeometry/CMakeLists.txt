project(DO_FeatureMatching_Examples)

message(STATUS "  - DO_FeatureMatching_Examples")

find_package(DO_Sara COMPONENTS
  Core Graphics ImageIO
  Features FeatureDetectors FeatureDescriptors
  KDTree Match FeatureMatching REQUIRED)

add_executable(essential_matrix_example
  essential_matrix_example.cpp
  sift.cpp sift.hpp)

set_target_properties(essential_matrix_example PROPERTIES
  COMPILE_FLAGS ${SARA_DEFINITIONS})
target_link_libraries(essential_matrix_example ${DO_Sara_LIBRARIES})

set_property(TARGET essential_matrix_example
  PROPERTY
  FOLDER "DO Sara Examples/MultiViewGeometry")
