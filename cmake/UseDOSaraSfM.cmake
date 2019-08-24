if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_SfM_ADDED GLOBAL PROPERTY _DO_Sara_SfM_INCLUDED)

  if (NOT DO_Sara_SfM_ADDED)
    #set(Boost_DEBUG ON)
    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_MULTITHREADED ON)
    find_package(Boost COMPONENTS filesystem system REQUIRED)

    sara_glob_directory(${DO_Sara_SOURCE_DIR}/SfM)
    sara_create_common_variables("SfM")
    sara_set_internal_dependencies("SfM"
      "Features;FeatureDetectors;FeatureDescriptors;FeatureMatching;MultiViewGeometry")
    sara_generate_library("SfM")

    target_include_directories(DO_Sara_SfM PRIVATE
      ${Boost_INCLUDE_DIR}
      ${DO_Sara_ThirdParty_DIR}/eigen
      ${DO_Sara_INCLUDE_DIR})
    target_compile_definitions(DO_Sara_SfM
      PRIVATE -DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB)
    target_link_libraries(DO_Sara_SfM tinyply)
  endif ()
endif ()
