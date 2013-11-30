macro (do_set_featurematching_source_dir)
  set(DO_FeatureMatching_SOURCE_DIR ${DO_SOURCE_DIR}/FeatureMatching)
endmacro (do_set_featurematching_source_dir)

macro (do_list_featurematching_source_files)
  # Master header file
  set(DO_FeatureMatching_MASTER_HEADER ${DO_SOURCE_DIR}/FeatureMatching.hpp)
  source_group("Master Header File" FILES ${DO_FeatureMatching_MASTER_HEADER})
  # Header files
  file(GLOB DO_FeatureMatching_HEADER_FILES
       ${DO_FeatureMatching_SOURCE_DIR}/*.hpp)
  # Source files
  file(GLOB DO_FeatureMatching_SOURCE_FILES
       ${DO_FeatureMatching_SOURCE_DIR}/*.cpp)
  # All header files here
  set(DO_FeatureMatching_HEADER_FILES
      ${DO_FeatureMatching_MASTER_HEADER}
      ${DO_FeatureMatching_HEADER_FILES})
endmacro (do_list_featurematching_source_files)

macro (do_load_packages_for_featurematching_library)
  include(${DO_Core_USE_FILE})
  include(${DO_Graphics_USE_FILE})
  include_directories(${flann_DIR}/src/cpp)
endmacro (do_load_packages_for_featurematching_library)

macro (do_create_variables_for_featurematching_library)
  set(DO_FeatureMatching_LIBRARIES DO_FeatureMatching)
  set(DO_FeatureMatching_LINK_LIBRARIES "${DO_Graphics_LIBRARIES};flann_cpp_s")
endmacro (do_create_variables_for_featurematching_library)

do_load_packages_for_featurematching_library()

if (DO_USE_FROM_SOURCE)
  get_property(DO_FeatureMatching_ADDED GLOBAL PROPERTY _DO_FeatureMatching_INCLUDED)
  if (NOT DO_FeatureMatching_ADDED)
    do_set_featurematching_source_dir()
    do_list_featurematching_source_files()
    do_create_variables_for_featurematching_library()
    
    # Static library
    do_append_library(
      FeatureMatching STATIC
      "${DO_SOURCE_DIR}"
      "${DO_FeatureMatching_HEADER_FILES}"
      "${DO_FeatureMatching_SOURCE_FILES}"
      "${DO_FeatureMatching_LINK_LIBRARIES}"
    )
    do_set_specific_target_properties(DO_FeatureMatching DO_STATIC)

    # Shared library
    if (DO_BUILD_SHARED_LIBS)
      do_append_library(
        FeatureMatching_SHARED SHARED
        "${DO_SOURCE_DIR}"
        "${DO_FeatureMatching_HEADER_FILES}"
        "${DO_FeatureMatching_SOURCE_FILES}"
        "${DO_FeatureMatching_LINK_LIBRARIES}"
      )
      do_set_specific_target_properties(DO_FeatureMatching DO_EXPORTS)
    endif ()    
  endif ()
endif ()
