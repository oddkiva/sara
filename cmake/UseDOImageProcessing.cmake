macro (do_set_imageprocessing_source_dir)
    set(DO_ImageProcessing_SOURCE_DIR ${DO_SOURCE_DIR}/ImageProcessing)
endmacro (do_set_imageprocessing_source_dir)

macro (do_list_imageprocessing_source_files)
    # Master header file
    set(DO_ImageProcessing_MASTER_HEADER ${DO_SOURCE_DIR}/ImageProcessing.hpp)
    source_group("Master Header File" FILES ${DO_ImageProcessing_MASTER_HEADER})
    # Header files
    file(GLOB DO_ImageProcessing_HEADER_FILES
         ${DO_ImageProcessing_SOURCE_DIR}/*.hpp)
    # Source files
    file(GLOB DO_ImageProcessing_SOURCE_FILES
         ${DO_ImageProcessing_SOURCE_DIR}/*.cpp)
    # All files here
    set(DO_ImageProcessing_HEADER_FILES
        ${DO_ImageProcessing_MASTER_HEADER}
        ${DO_ImageProcessing_HEADER_FILES})
endmacro (do_list_imageprocessing_source_files)

macro (do_load_packages_for_imageprocessing_library)
    include(${DO_Core_USE_FILE})
endmacro (do_load_packages_for_imageprocessing_library)

macro (do_create_variables_for_imageprocessing_library)
    set(DO_ImageProcessing_LIBRARIES DO_ImageProcessing)
    set(DO_ImageProcessing_LINK_LIBRARIES ${DO_ImageProcessing_LIBRARIES})
endmacro (do_create_variables_for_imageprocessing_library)


do_load_packages_for_imageprocessing_library()


if (DO_USE_FROM_SOURCE)
    get_property(DO_ImageProcessing_ADDED GLOBAL PROPERTY _DO_ImageProcessing_INCLUDED)
    if (NOT DO_ImageProcessing_ADDED)
        do_set_imageprocessing_source_dir()
        do_list_imageprocessing_source_files()
        do_create_variables_for_imageprocessing_library()
    endif ()
    
    # Static library
    do_append_library(
        ImageProcessing STATIC
        "${DO_SOURCE_DIR}"
        "${DO_ImageProcessing_HEADER_FILES}"
        "${DO_ImageProcessing_SOURCE_FILES}"
        "${DO_ImageProcessing_LINK_LIBRARIES}"
    )
    
    # Shared library
    if (DO_BUILD_SHARED_LIBS)
        do_append_library(
            ImageProcessing_SHARED SHARED
            "${DO_SOURCE_DIR}"
            "${DO_ImageProcessing_HEADER_FILES}"
            "${DO_ImageProcessing_SOURCE_FILES}"
            "${DO_ImageProcessing_LINK_LIBRARIES}"
        )
    endif ()
endif ()