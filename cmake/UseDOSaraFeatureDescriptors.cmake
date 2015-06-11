include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_Sara_ThirdParty_DIR}/eigen)

if (DO_USE_FROM_SOURCE)
  get_property(DO_Sara_FeatureDescriptors_ADDED GLOBAL PROPERTY
               _DO_Sara_FeaturesDescriptors_INCLUDED)
  if (NOT DO_Sara_FeaturesDescriptors_ADDED)
    do_glob_directory(${DO_Sara_SOURCE_DIR}/FeatureDescriptors)
    do_create_common_variables("FeatureDescriptors")
    do_set_internal_dependencies("FeatureDescriptors" "Core;Features;Graphics")
    do_generate_library("FeaturesDescriptors")
  endif ()
endif ()
