include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_Sara_ThirdParty_DIR}/eigen)

if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_DisjointSets_ADDED GLOBAL PROPERTY _DO_Sara_DisjointSets_INCLUDED)
  if (NOT DO_Sara_DisjointSets_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/DisjointSets)
    sara_create_common_variables("DisjointSets")
    sara_generate_library("DisjointSets")
  endif ()
endif ()
