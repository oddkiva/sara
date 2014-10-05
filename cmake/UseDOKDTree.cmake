include_directories(
  ${DO_INCLUDE_DIR}
  ${DO_ThirdParty_DIR}/eigen
  ${DO_ThirdParty_DIR}/flann/src/cpp)

if (DO_USE_FROM_SOURCE)
  get_property(DO_KDTree_ADDED GLOBAL PROPERTY _DO_KDTree_INCLUDED)
  if (NOT DO_KDTree_ADDED)
    do_glob_directory(${DO_SOURCE_DIR}/KDTree)
    do_create_common_variables("KDTree")
    do_generate_library("KDTree")
    target_link_libraries(DO_KDTree flann_cpp_s)
  endif ()
endif ()
