macro (do_set_featuredescriptors_source_dir)
    set(DO_FeatureDescriptors_SOURCE_DIR ${DO_SOURCE_DIR}/FeatureDescriptors)
endmacro (do_set_featuredescriptors_source_dir)

macro (do_list_featuredescriptors_source_files)
    # Master header file
    set(DO_FeatureDescriptors_MASTER_HEADER ${DO_SOURCE_DIR}/FeatureDescriptors.hpp)
    source_group("Master Header File" FILES ${DO_FeatureDescriptors_MASTER_HEADER})
    # Header files
    file(GLOB DO_FeatureDescriptors_HEADER_FILES
         ${DO_FeatureDescriptors_SOURCE_DIR}/*.hpp)
    # Source files
    file(GLOB DO_FeatureDescriptors_SOURCE_FILES
         ${DO_FeatureDescriptors_SOURCE_DIR}/*.cpp)
    # All header files here
    set(DO_FeatureDescriptors_HEADER_FILES
        ${DO_FeatureDescriptors_MASTER_HEADER}
        ${DO_FeatureDescriptors_HEADER_FILES})
    # Organize source files as follows.
    source_group("Dominant Orientation" FILES
                 ${DO_FeatureDescriptors_SOURCE_DIR}/Orientation.hpp
                 ${DO_FeatureDescriptors_SOURCE_DIR}/Orientation.cpp)
    source_group("SIFT Descriptor" FILES
                 ${DO_FeatureDescriptors_SOURCE_DIR}/SIFT.hpp
                 ${DO_FeatureDescriptors_SOURCE_DIR}/SIFT.cpp)
endmacro (do_list_featuredescriptors_source_files)


macro (do_load_packages_for_featuredescriptors_library)
    include(${DO_Core_USE_FILE})
    include(${DO_Graphics_USE_FILE})
    include(${DO_Features_USE_FILE})
    include_directories(${DO_ThirdParty_DIR})
endmacro (do_load_packages_for_featuredescriptors_library)

macro (do_create_variables_for_featuredescriptors_library)
    set(DO_FeatureDescriptors_LIBRARIES DO_FeatureDescriptors)
    set(DO_FeatureDescriptors_LINK_LIBRARIES
        ${DO_Features_LIBRARIES} ${DO_Graphics_LIBRARIES})
endmacro (do_create_variables_for_featuredescriptors_library)

do_load_packages_for_featuredescriptors_library()

if (DO_USE_FROM_SOURCE)
    get_property(DO_FeatureDescriptors_ADDED GLOBAL PROPERTY _DO_FeatureDescriptors_INCLUDED)
    if (NOT DO_FeatureDescriptors_ADDED)
        do_set_featuredescriptors_source_dir()
        do_list_featuredescriptors_source_files()
        do_create_variables_for_featuredescriptors_library()
    endif ()
    
    # Static library
    do_append_library(
        FeatureDescriptors STATIC
        "${DO_SOURCE_DIR}"
        "${DO_FeatureDescriptors_HEADER_FILES}"
        "${DO_FeatureDescriptors_SOURCE_FILES}"
        "${DO_FeatureDescriptors_LINK_LIBRARIES}"
    )
    do_set_specific_target_properties(DO_FeatureDescriptors DO_STATIC)
      
    # Shared library
    if (DO_BUILD_SHARED_LIBS)
      do_append_library(
        FeatureDescriptors_SHARED SHARED
        "${DO_SOURCE_DIR}"
        "${DO_FeatureDescriptors_HEADER_FILES}"
        "${DO_FeatureDescriptors_SOURCE_FILES}"
        "${DO_FeatureDescriptors_LINK_LIBRARIES}"
      )
      do_set_specific_target_properties(DO_FeatureDescriptors DO_EXPORTS)
    endif ()
endif()
