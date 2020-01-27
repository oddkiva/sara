add_executable(evaluate_on_mikolajczyk evaluate_on_mikolajczyk.cpp)
target_link_libraries(evaluate_on_mikolajczyk
  ${DO_Sara_LIBRARIES}
  DO::Sara::EmpiricalValidation
  DO::Sara::HyperParameterLearning
  DO::Sara::MatchPropagation)
set_property(TARGET evaluate_on_mikolajczyk PROPERTY
  FOLDER "DO Sara Tools/MatchPropagation")

add_executable(evaluate_region evaluate_region.cpp)
target_link_libraries(evaluate_region
  ${DO_Sara_LIBRARIES}
  DO::Sara::EmpiricalValidation
  DO::Sara::HyperParameterLearning
  DO::Sara::MatchPropagation)
set_property(TARGET evaluate_region PROPERTY
  FOLDER "DO Sara Tools/MatchPropagation")

add_executable(evaluate_region_growing evaluate_region_growing.cpp)
target_link_libraries(evaluate_region_growing
  ${DO_Sara_LIBRARIES}
  DO::Sara::EmpiricalValidation
  DO::Sara::HyperParameterLearning
  DO::Sara::MatchPropagation)
set_property(TARGET evaluate_region_growing PROPERTY
  FOLDER "DO Sara Tools/MatchPropagation")

add_executable(validate_region_growing_neighborhood_params validate_region_growing_neighborhood_params.cpp)
target_link_libraries(validate_region_growing_neighborhood_params
  ${DO_Sara_LIBRARIES}
  DO::Sara::EmpiricalValidation
  DO::Sara::HyperParameterLearning
  DO::Sara::MatchPropagation)
set_property(TARGET validate_region_growing_neighborhood_params PROPERTY
  FOLDER "DO Sara Tools/MatchPropagation")

