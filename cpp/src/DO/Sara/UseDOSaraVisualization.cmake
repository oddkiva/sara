if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_Visualization_ADDED GLOBAL PROPERTY _DO_Sara_Visualization_INCLUDED)
  if (NOT DO_Sara_Visualization_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/Visualization)
    sara_create_common_variables("Visualization")
    sara_generate_library("Visualization")
    target_link_libraries(DO_Sara_Visualization
      PRIVATE
      DO::Sara::Core
      DO::Sara::Graphics
      DO::Sara::Features
      DO::Sara::Match)
  endif()
endif ()
