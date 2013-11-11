macro (do_set_match_source_dir)
  set(DO_Match_SOURCE_DIR ${DO_SOURCE_DIR}/Match)
endmacro (do_set_match_source_dir)

macro (do_list_match_source_files)
  # Master header file
  set(DO_Match_MASTER_HEADER ${DO_SOURCE_DIR}/Match.hpp)
  source_group("Master Header File" FILES ${DO_Match_MASTER_HEADER})
  # Header files
  file(GLOB DO_Match_HEADER_FILES
       ${DO_Match_SOURCE_DIR}/*.hpp)
  # Source files
  file(GLOB DO_Match_SOURCE_FILES
       ${DO_Match_SOURCE_DIR}/*.cpp)
  # All header files here
  set(DO_Match_HEADER_FILES
      ${DO_Match_MASTER_HEADER}
      ${DO_Match_HEADER_FILES})
endmacro (do_list_match_source_files)

macro (do_load_packages_for_match_library)
  include(${DO_Core_USE_FILE})
  include(${DO_Graphics_USE_FILE})
endmacro (do_load_packages_for_match_library)

macro (do_create_variables_for_match_library)
  set(DO_Match_LIBRARIES DO_Match)
  set(DO_Match_LINK_LIBRARIES ${DO_Graphics_LIBRARIES})
endmacro (do_create_variables_for_match_library)

do_load_packages_for_match_library()

if (DO_USE_FROM_SOURCE)
  get_property(DO_Match_ADDED GLOBAL PROPERTY _DO_Match_INCLUDED)
  if (NOT DO_Match_ADDED)
    do_set_match_source_dir()
    do_list_match_source_files()
    do_create_variables_for_match_library()
  endif ()

  # Static library
  do_append_library(
    Match STATIC
    "${DO_SOURCE_DIR}"
    "${DO_Match_HEADER_FILES}"
    "${DO_Match_SOURCE_FILES}"
    "${DO_Match_LINK_LIBRARIES}"
  )
  do_set_specific_target_properties(DO_Match DO_STATIC)
    
  # Shared library
  if (DO_BUILD_SHARED_LIBS)
    do_append_library(
      Match_SHARED SHARED
      "${DO_SOURCE_DIR}"
      "${DO_Match_HEADER_FILES}"
      "${DO_Match_SOURCE_FILES}"
      "${DO_Match_LINK_LIBRARIES}"
    )
    do_set_specific_target_properties(DO_Match DO_EXPORTS)
  endif ()
endif ()