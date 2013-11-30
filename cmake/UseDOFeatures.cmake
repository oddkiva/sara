macro (do_set_features_source_dir)
  set(DO_Features_SOURCE_DIR ${DO_SOURCE_DIR}/Features)
endmacro (do_set_features_source_dir)

macro (do_list_features_source_files)
  # Master header file
  set(DO_Features_MASTER_HEADER ${DO_SOURCE_DIR}/Features.hpp)
  source_group("Master Header File" FILES ${DO_Features_MASTER_HEADER})
  # Header files
  file(GLOB DO_Features_HEADER_FILES
       ${DO_Features_SOURCE_DIR}/*.hpp)
  # Source files
  file(GLOB DO_Features_SOURCE_FILES
       ${DO_Features_SOURCE_DIR}/*.cpp)
  # All header files here
  set(DO_Features_HEADER_FILES
      ${DO_Features_MASTER_HEADER}
      ${DO_Features_HEADER_FILES})
endmacro (do_list_features_source_files)

macro (do_load_packages_for_features_library)
  include(${DO_Core_USE_FILE})
  include(${DO_Graphics_USE_FILE})
endmacro (do_load_packages_for_features_library)

macro (do_create_variables_for_features_library)
  set(DO_Features_LIBRARIES DO_Features)
  set(DO_Features_LINK_LIBRARIES ${DO_Graphics_LIBRARIES})
endmacro (do_create_variables_for_features_library)

do_load_packages_for_features_library()

if (DO_USE_FROM_SOURCE)
  get_property(DO_Features_ADDED GLOBAL PROPERTY _DO_Features_INCLUDED)
  if (NOT DO_Features_ADDED)
    do_set_features_source_dir()
    do_list_features_source_files()
    do_create_variables_for_features_library()
    
    # Static library
    do_append_library(
      Features STATIC
      "${DO_SOURCE_DIR}"
      "${DO_Features_HEADER_FILES}"
      "${DO_Features_SOURCE_FILES}"
      "${DO_Features_LINK_LIBRARIES}"
    )
    do_set_specific_target_properties(DO_Features DO_STATIC)

    # Shared library
    if (DO_BUILD_SHARED_LIBS)
      do_append_library(
        Features_SHARED SHARED
        "${DO_SOURCE_DIR}"
        "${DO_Features_HEADER_FILES}"
        "${DO_Features_SOURCE_FILES}"
        "${DO_Features_LINK_LIBRARIES}"
      )
      do_set_specific_target_properties(DO_Features DO_EXPORTS)
    endif ()
  endif ()
endif ()
