if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_FileSystem_ADDED GLOBAL PROPERTY _DO_Sara_FileSystem_INCLUDED)

  if (NOT DO_Sara_FileSystem_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/FileSystem)
    sara_create_common_variables("FileSystem")
    sara_generate_library("FileSystem")

    target_include_directories(DO_Sara_FileSystem
      PRIVATE
      $<BUILD_INTERFACE:${DO_Sara_ThirdParty_DIR}>
      $<BUILD_INTERFACE:${DO_Sara_INCLUDE_DIR}>)
    target_compile_definitions(DO_Sara_FileSystem
      PUBLIC
      -DBOOST_ALL_DYN_LINK
      -DBOOST_ALL_NO_LIB)

    target_link_libraries(DO_Sara_FileSystem PRIVATE Boost::filesystem)
  endif ()
endif ()
