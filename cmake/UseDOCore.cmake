include_directories(
  ${DO_INCLUDE_DIR}
  ${DO_ThirdParty_DIR}/eigen)

if (DO_USE_FROM_SOURCE)
  get_property(DO_Core_ADDED GLOBAL PROPERTY _DO_Core_INCLUDED)
  if (NOT DO_Core_ADDED)
    do_glob_directory(${DO_SOURCE_DIR}/Core)
    do_create_common_variables("Core")
    do_generate_library("Core")
  endif ()
endif ()