find_package(HDF5 COMPONENTS CXX REQUIRED)

if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_Core_ADDED GLOBAL PROPERTY _DO_Sara_Core_INCLUDED)

  if (NOT DO_Sara_Core_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/Core)
    sara_create_common_variables("Core")
    sara_generate_library("Core")

    target_include_directories(DO_Sara_Core
      PRIVATE ${DO_Sara_INCLUDE_DIR}
              ${DO_Sara_ThirdParty_DIR}/eigen
      PUBLIC ${HDF5_INCLUDE_DIRS})

  endif ()
endif ()
