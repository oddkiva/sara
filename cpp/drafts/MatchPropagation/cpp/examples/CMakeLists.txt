find_package(DO_Sara COMPONENTS FeatureDetectors FeatureDescriptors REQUIRED)

add_definitions(-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR})
add_executable(example_match_propagation main.cpp)
target_link_libraries(example_match_propagation
  ${DO_Sara_LIBRARIES}
  DO::Sara::MatchPropagation)
set_property(TARGET example_match_propagation
  PROPERTY
  FOLDER "Examples/Sara/MatchPropagation")
