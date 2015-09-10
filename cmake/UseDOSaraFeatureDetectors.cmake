include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_Sara_ThirdParty_DIR}/eigen)

if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_FeatureDetectors_ADDED GLOBAL PROPERTY _DO_Sara_FeatureDetectors_INCLUDED)
  if (NOT DO_Sara_FeatureDetectors_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/FeatureDetectors)
    sara_create_common_variables("FeatureDetectors")
    sara_set_internal_dependencies("FeatureDetectors" "Core;Features;Graphics")
    sara_generate_library("FeatureDetectors")
  endif ()
endif ()
