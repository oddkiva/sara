if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_MultiViewGeometry_ADDED GLOBAL PROPERTY _DO_Sara_MultiViewGeometry_INCLUDED)
  if (NOT DO_Sara_MultiViewGeometry_ADDED)
    find_package(HDF5 COMPONENTS CXX REQUIRED)

    sara_glob_directory(${DO_Sara_SOURCE_DIR}/MultiViewGeometry)
    sara_create_common_variables("MultiViewGeometry")
    sara_set_internal_dependencies("MultiViewGeometry"
      "Core;Features;FileSystem;ImageIO")
    sara_generate_library("MultiViewGeometry")

    target_include_directories(DO_Sara_MultiViewGeometry
      PUBLIC ${Boost_INCLUDE_DIR}
             ${DO_Sara_INCLUDE_DIR}
             ${DO_Sara_ThirdParty_DIR}/eigen)
    target_include_directories(DO_Sara_MultiViewGeometry PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(DO_Sara_MultiViewGeometry PUBLIC ${HDF5_LIBRARIES})
  endif ()
endif ()
