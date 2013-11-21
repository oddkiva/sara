macro (do_set_kdtree_source_dir)
    set(DO_KDTree_SOURCE_DIR ${DO_SOURCE_DIR}/KDTree)
endmacro (do_set_kdtree_source_dir)

macro (do_list_kdtree_source_files)
    # Master header file
    set(DO_KDTree_MASTER_HEADER ${DO_SOURCE_DIR}/KDTree.hpp)
    source_group("Master Header File" FILES ${DO_KDTree_MASTER_HEADER})
    # Header files
    file(GLOB DO_KDTree_HEADER_FILES
         ${DO_KDTree_SOURCE_DIR}/*.hpp)
    # Source files
    file(GLOB DO_KDTree_SOURCE_FILES
         ${DO_KDTree_SOURCE_DIR}/*.cpp)
    # All header files here
    set(DO_KDTree_HEADER_FILES
        ${DO_KDTree_MASTER_HEADER}
        ${DO_KDTree_HEADER_FILES})
endmacro (do_list_kdtree_source_files)


macro (do_load_packages_for_kdtree_library)
    include_directories(${flann_DIR}/src/cpp)
endmacro (do_load_packages_for_kdtree_library)

macro (do_create_variables_for_kdtree_library)
    set(DO_KDTree_LIBRARIES DO_KDTree)
    set(DO_KDTree_LINK_LIBRARIES flann_cpp_s)
endmacro (do_create_variables_for_kdtree_library)

do_load_packages_for_kdtree_library()

if (DO_USE_FROM_SOURCE)
  get_property(DO_KDTree_ADDED GLOBAL PROPERTY _DO_KDTree_INCLUDED)
  if (NOT DO_KDTree_ADDED)
    do_set_kdtree_source_dir()
    do_list_kdtree_source_files()
    do_create_variables_for_kdtree_library()

    # Static library
    do_append_library(
        KDTree STATIC
        "${DO_SOURCE_DIR}"
        "${DO_KDTree_HEADER_FILES}"
        "${DO_KDTree_SOURCE_FILES}"
        "${DO_KDTree_LINK_LIBRARIES}"
    )
    do_set_specific_target_properties(DO_KDTree DO_STATIC)
    do_cotire(KDTree ${DO_KDTree_MASTER_HEADER})

    # Shared library
    if (DO_BUILD_SHARED_LIBS)
      do_append_library(
        KDTree_SHARED SHARED
        "${DO_SOURCE_DIR}"
        "${DO_KDTree_HEADER_FILES}"
        "${DO_KDTree_SOURCE_FILES}"
        "${DO_KDTree_LINK_LIBRARIES}"
      )
      do_set_specific_target_properties(DO_KDTree DO_EXPORTS)
    endif ()
  endif ()    
endif()
