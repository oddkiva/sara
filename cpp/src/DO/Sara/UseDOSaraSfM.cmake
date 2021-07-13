if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_SfM_ADDED GLOBAL PROPERTY _DO_Sara_SfM_INCLUDED)

  if (NOT DO_Sara_SfM_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/SfM)
    sara_create_common_variables("SfM")
    sara_generate_library("SfM")

    target_include_directories(DO_Sara_SfM
      PRIVATE
      ${DO_Sara_ThirdParty_DIR}/eigen
      ${DO_Sara_INCLUDE_DIR})
    target_compile_definitions(DO_Sara_SfM
      PRIVATE
      BOOST_ALL_DYN_LINK
      BOOST_ALL_NO_LIB)
    target_link_libraries(DO_Sara_SfM
      PRIVATE
      tinyply
      Boost::filesystem
      PUBLIC
      DO::Sara::Features
      DO::Sara::FeatureDetectors
      DO::Sara::FeatureDescriptors
      DO::Sara::FeatureMatching
      DO::Sara::MultiViewGeometry
      DO::Sara::Visualization
      $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)
  endif ()
endif ()
