if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_Features_ADDED GLOBAL PROPERTY _DO_Sara_Features_INCLUDED)

  if (NOT DO_Sara_Features_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/Features)
    sara_create_common_variables("Features")
    sara_generate_library("Features")

    target_include_directories(DO_Sara_Features
      PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(DO_Sara_Features
      PRIVATE DO::Sara::Geometry
      PUBLIC Eigen3::Eigen
             ${HDF5_LIBRARIES})
  endif ()
endif ()
