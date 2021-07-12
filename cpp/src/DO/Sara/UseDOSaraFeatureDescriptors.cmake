if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_FeatureDescriptors_ADDED GLOBAL PROPERTY
               _DO_Sara_FeatureDescriptors_INCLUDED)
  if (NOT DO_Sara_FeatureDescriptors_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/FeatureDescriptors)
    sara_create_common_variables("FeatureDescriptors")
    sara_set_internal_dependencies("FeatureDescriptors" "Core;Features")
    sara_generate_library("FeatureDescriptors")

    target_link_libraries(DO_Sara_FeatureDescriptors
      PUBLIC
      $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)
  endif ()
endif ()
