include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_Sara_ThirdParty_DIR}/eigen)

if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_MultiViewGeometry_ADDED GLOBAL PROPERTY _DO_Sara_MultiViewGeometry_INCLUDED)
  if (NOT DO_Sara_MultiViewGeometry_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/SfM)
    sara_create_common_variables("SfM")
    sara_set_internal_dependencies("SfM" "Core")
    sara_generate_library("SfM")
  endif ()
endif ()
