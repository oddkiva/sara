include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_Sara_ThirdParty_DIR}/eigen
  ${DO_Sara_ThirdParty_DIR}/flann/src/cpp)

if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_FeatureMatching_ADDED GLOBAL PROPERTY
               _DO_Sara_FeatureMatching_INCLUDED)
  if (NOT DO_Sara_FeatureMatching_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/FeatureMatching)
    sara_create_common_variables("FeatureMatching")
    sara_set_internal_dependencies(
      "FeatureMatching"
      "Core;Features;Graphics;KDTree;Match")
    sara_generate_library("FeatureMatching")
  endif ()
endif ()
