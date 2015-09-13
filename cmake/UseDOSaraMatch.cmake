include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_Sara_ThirdParty_DIR}/eigen)

if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_Match_ADDED GLOBAL PROPERTY _DO_Sara_Match_INCLUDED)
  if (NOT DO_Sara_Match_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/Match)
    sara_create_common_variables("Match")
    sara_set_internal_dependencies(
      "Match"
      "Core;Features;Graphics")
    sara_generate_library("Match")
  endif ()
endif ()
