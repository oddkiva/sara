include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_Sara_ThirdParty_DIR}/eigen)

if (DO_USE_FROM_SOURCE)
  get_property(DO_Sara_Match_ADDED GLOBAL PROPERTY _DO_Sara_Match_INCLUDED)
  if (NOT DO_Sara_Match_ADDED)
    do_glob_directory(${DO_Sara_SOURCE_DIR}/Match)
    do_create_common_variables("Match")
    do_generate_library("Match")
  endif ()
endif ()
