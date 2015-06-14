include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_ThirdParty_DIR}/eigen)

if (DO_USE_FROM_SOURCE)
  get_property(DO_Sara_ImageProcessing_ADDED GLOBAL PROPERTY _DO_Sara_ImageProcessing_INCLUDED)
  if (NOT DO_Sara_ImageProcessing_ADDED)
    do_glob_directory(${DO_Sara_SOURCE_DIR}/ImageProcessing)
    do_create_common_variables("ImageProcessing")
    do_generate_library("ImageProcessing")
  endif ()
endif ()
