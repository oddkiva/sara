macro (do_set_geometry_source_dir)
  set(DO_Geometry_SOURCE_DIR ${DO_SOURCE_DIR}/Geometry)
endmacro (do_set_geometry_source_dir)

macro (do_list_geometry_source_files)
  # Master header file
  set(DO_Geometry_MASTER_HEADER ${DO_SOURCE_DIR}/Geometry.hpp)
  source_group("Master Header File" FILES ${DO_Geometry_MASTER_HEADER})
  # Header files
  file(GLOB DO_Geometry_HEADER_FILES
       ${DO_Geometry_SOURCE_DIR}/*.hpp)
  # Source files
  file(GLOB DO_Geometry_SOURCE_FILES
       ${DO_Geometry_SOURCE_DIR}/*.cpp)
  # All header files here
  set(DO_Geometry_HEADER_FILES
      ${DO_Geometry_MASTER_HEADER}
      ${DO_Geometry_HEADER_FILES})
endmacro (do_list_geometry_source_files)

macro (do_load_packages_for_geometry_library)
  find_package(Boost 1.47 REQUIRED)
  include_directories(${Boost_INCLUDE_DIRS})
  include(${DO_Core_USE_FILE})
  include(${DO_Graphics_USE_FILE})
endmacro (do_load_packages_for_geometry_library)

macro (do_create_variables_for_geometry_library)
  set(DO_Geometry_LIBRARIES DO_Geometry)
  set(DO_Geometry_LINK_LIBRARIES ${DO_Graphics_LIBRARIES})
endmacro (do_create_variables_for_geometry_library)

do_load_packages_for_geometry_library()

if (DO_USE_FROM_SOURCE)
  get_property(DO_Geometry_ADDED GLOBAL PROPERTY _DO_Geometry_INCLUDED)
  if (NOT DO_Geometry_ADDED)
    do_set_geometry_source_dir()
    do_list_geometry_source_files()
    do_create_variables_for_geometry_library()
  endif ()

  # Static library
  do_append_library(
    Geometry STATIC
    "${DO_SOURCE_DIR}"
    "${DO_Geometry_HEADER_FILES}"
    "${DO_Geometry_SOURCE_FILES}"
    "${DO_Geometry_LINK_LIBRARIES}"
  )
  do_set_specific_target_properties(DO_Geometry DO_STATIC)
    
  # Shared library
  if (DO_BUILD_SHARED_LIBS)
    do_append_library(
      Geometry_SHARED SHARED
      "${DO_SOURCE_DIR}"
      "${DO_Geometry_HEADER_FILES}"
      "${DO_Geometry_SOURCE_FILES}"
      "${DO_Geometry_LINK_LIBRARIES}"
    )
    do_set_specific_target_properties(DO_Geometry DO_EXPORTS)
  endif ()
endif ()