if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_FeatureDescriptors_ADDED GLOBAL PROPERTY
               _DO_Sara_FeatureDescriptors_INCLUDED)
  if (NOT DO_Sara_FeatureDescriptors_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/FeatureDescriptors)
    sara_create_common_variables("FeatureDescriptors")
    sara_set_internal_dependencies("FeatureDescriptors" "Core;Features;Graphics")
    sara_generate_library("FeatureDescriptors")

    target_include_directories(DO_Sara_FeatureDescriptors PRIVATE
      ${DO_Sara_INCLUDE_DIR}
      ${DO_Sara_ThirdParty_DIR}/eigen)

  endif ()
endif ()
