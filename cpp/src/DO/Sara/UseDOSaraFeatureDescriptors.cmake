if(SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_FeatureDescriptors_ADDED GLOBAL
               PROPERTY _DO_Sara_FeatureDescriptors_INCLUDED)
  if(NOT DO_Sara_FeatureDescriptors_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/FeatureDescriptors)
    sara_create_common_variables("FeatureDescriptors")
    sara_generate_library("FeatureDescriptors")

    if(SARA_USE_HALIDE)
      target_compile_definitions(DO_Sara_FeatureDescriptors
                                 PRIVATE DO_SARA_USE_HALIDE)
    endif ()
    target_link_libraries(
      DO_Sara_FeatureDescriptors
      PUBLIC Eigen3::Eigen #
             DO::Sara::Features #
             DO::Sara::ImageProcessing
             $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
  endif()
endif()
