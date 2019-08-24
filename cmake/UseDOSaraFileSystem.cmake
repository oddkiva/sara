if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_FileSystem_ADDED GLOBAL PROPERTY _DO_Sara_FileSystem_INCLUDED)

  if (NOT DO_Sara_FileSystem_ADDED)
	#set(Boost_DEBUG ON)
    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_MULTITHREADED ON)
    find_package(Boost COMPONENTS filesystem system REQUIRED)

    sara_glob_directory(${DO_Sara_SOURCE_DIR}/FileSystem)
    sara_create_common_variables("FileSystem")
    sara_generate_library("FileSystem")

    target_include_directories(DO_Sara_FileSystem PRIVATE
      ${Boost_INCLUDE_DIR}
      ${DO_Sara_INCLUDE_DIR})
    target_compile_definitions(DO_Sara_FileSystem
      PRIVATE -DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB)

    target_link_libraries(DO_Sara_FileSystem ${Boost_LIBRARIES})
  endif ()
endif ()
