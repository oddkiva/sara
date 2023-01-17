if(SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_RANSAC_ADDED GLOBAL PROPERTY _DO_Sara_RANSAC_INCLUDED)

  if(NOT DO_Sara_RANSAC_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/RANSAC)
    sara_create_common_variables("RANSAC")
    sara_generate_library("RANSAC")
  endif()
endif()
