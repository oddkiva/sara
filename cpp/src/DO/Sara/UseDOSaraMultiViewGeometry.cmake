if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_MultiViewGeometry_ADDED GLOBAL PROPERTY _DO_Sara_MultiViewGeometry_INCLUDED)
  if (NOT DO_Sara_MultiViewGeometry_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/MultiViewGeometry)
    sara_create_common_variables("MultiViewGeometry")
    sara_generate_library("MultiViewGeometry")

    target_include_directories(DO_Sara_MultiViewGeometry
      PUBLIC
      ${Boost_INCLUDE_DIRS})
    target_link_libraries(DO_Sara_MultiViewGeometry
      PUBLIC
      DO::Sara::Core
      DO::Sara::Features
      DO::Sara::FileSystem
      DO::Sara::ImageIO)
  endif ()
endif ()
