project(DO_MultiViewGeometry_Examples)

message(STATUS "  - DO_MultiViewGeometry_Examples")

find_package(DO_Sara COMPONENTS
  Core Graphics ImageIO Features FeatureDetectors FeatureDescriptors KDTree
  Match FeatureMatching MultiViewGeometry SfM REQUIRED)
if (NOT WIN32)
  find_package(Ceres REQUIRED)
endif ()


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


if (WIN32)
  target_include_directories(essential_matrix_estimation_example
    PRIVATE ${DO_Sara_ThirdParty_DIR}/ceres-solver-1.14.0/include
            ${DO_Sara_ThirdParty_DIR}/ceres-solver-1.14.0/internal/ceres/miniglog
            ${DO_Sara_ThirdParty_DIR}/ceres-solver-1.14.0/config)
  target_link_libraries(essential_matrix_estimation_example
    PRIVATE ceres)
  target_compile_definitions(essential_matrix_estimation_example
    PRIVATE -DCERES_USING_SHARED_LIBRARY
            -DCERES_USE_OPENMP
            -DMAX_LOG_LEVEL=2)
else ()
  target_include_directories(essential_matrix_estimation_example
    PRIVATE ${CERES_INCLUDE_DIRS})
  target_link_libraries(essential_matrix_estimation_example
    PRIVATE ${CERES_LIBRARIES})
endif ()
