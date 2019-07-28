include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_Sara_ThirdParty_DIR}/eigen)

if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_SfM_ADDED GLOBAL PROPERTY _DO_Sara_SfM_INCLUDED)
  if (NOT DO_Sara_SfM_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/SfM)
    sara_create_common_variables("SfM")
    sara_set_internal_dependencies("SfM"
      "Features;FeatureDetectors;FeatureDescriptors;FeatureMatching;MultiViewGeometry")
    sara_generate_library("SfM")
    target_link_libraries(DO_Sara_SfM tinyply)
  endif ()
endif ()
