if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_Geometry_ADDED GLOBAL PROPERTY _DO_Sara_Geometry_INCLUDED)
  if (NOT DO_Sara_Geometry_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/Geometry)
    sara_create_common_variables("Geometry")
    sara_generate_library("Geometry")
    target_link_libraries(DO_Sara_Geometry PUBLIC DO::Sara::Core)
  endif ()
endif ()
