if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_KDTree_ADDED GLOBAL PROPERTY _DO_Sara_KDTree_INCLUDED)

  if (NOT DO_Sara_KDTree_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/KDTree)
    sara_create_common_variables("KDTree")
    sara_generate_library("KDTree")

    target_include_directories(DO_Sara_KDTree
      PUBLIC
      $<BUILD_INTERFACE:${DO_Sara_ThirdParty_DIR}/flann/src/cpp>)
    target_link_libraries(DO_Sara_KDTree
      PRIVATE flann_cpp_s
      PUBLIC $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)
  endif ()
endif ()
