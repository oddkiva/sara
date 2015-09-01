include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_Sara_ThirdParty_DIR}/eigen
  ${DO_Sara_ThirdParty_DIR}/flann/src/cpp)

if (DO_USE_FROM_SOURCE)
  get_property(DO_Sara_FeatureMatching_ADDED GLOBAL PROPERTY
               _DO_Sara_FeatureMatching_INCLUDED)
  if (NOT DO_Sara_FeatureMatching_ADDED)
    do_glob_directory(${DO_Sara_SOURCE_DIR}/FeatureMatching)
    do_create_common_variables("FeatureMatching")
    do_set_internal_dependencies(
      "FeatureMatching"
      "Core;Features;Graphics;KDTree")
    do_generate_library("FeatureMatching")
  endif ()
endif ()
