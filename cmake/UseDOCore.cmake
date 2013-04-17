macro (do_set_core_source_dir)
    set(DO_Core_SOURCE_DIR ${DO_SOURCE_DIR}/Core)
endmacro()

macro (do_list_core_source_files)
    # Master header file
    set(DO_Core_MASTER_HEADER ${DO_SOURCE_DIR}/Core.hpp)
    source_group("Master Header File" FILES ${DO_Core_MASTER_HEADER})
    # Template meta-programming stuff
    set(DO_Core_METAPROGRAMMING
        ${DO_Core_SOURCE_DIR}/StaticAssert.hpp
        ${DO_Core_SOURCE_DIR}/Meta.hpp)
    source_group("Meta-Programming Header Files" FILES ${DO_Core_METAPROGRAMMING})
    # Miscellaneous stuff
    set(DO_Core_MISC
        ${DO_Core_SOURCE_DIR}/Stringify.hpp
        ${DO_Core_SOURCE_DIR}/Timer.hpp)
    source_group("Misc Header Files" FILES ${DO_Core_MISC})
    # Linear algebra by integrating the 'Eigen' library
    set(DO_Core_EIGEN
        ${DO_Core_SOURCE_DIR}/EigenExtension.hpp)
    source_group("Eigen Integration Header Files" FILES ${DO_Core_EIGEN})
    # N-dimensional array with N-dimensional iterators
    set(DO_Core_MULTIARRAY
        ${DO_Core_SOURCE_DIR}/Locator.hpp
        ${DO_Core_SOURCE_DIR}/MultiArray.hpp
        ${DO_Core_SOURCE_DIR}/SparseMultiArray.hpp)
    source_group("Multi-Array Header Files" FILES ${DO_Core_MULTIARRAY})
    # Image and color data structures
    set(DO_Core_IMAGE
        ${DO_Core_SOURCE_DIR}/Color.hpp
        ${DO_Core_SOURCE_DIR}/Image.hpp)
    source_group("Image Header Files" FILES ${DO_Core_IMAGE})
    # Tree data structure
    set(DO_Core_TREE
        ${DO_Core_SOURCE_DIR}/Tree.hpp)
    source_group("Tree Header Files" FILES ${DO_Core_TREE})
    
    # All files here
    set(DO_Core_FILES
        ${DO_Core_MASTER_HEADER}
        ${DO_Core_METAPROGRAMMING}
        ${DO_Core_MISC}
        ${DO_Core_EIGEN}
        ${DO_Core_MULTIARRAY}
        ${DO_Core_IMAGE}
        ${DO_Core_TREE})
endmacro (do_list_core_source_files)

macro (do_create_variables_for_core_library)
    set(DO_Core_LIBRARIES "")
    set(DO_Core_LINK_LIBRARIES "")
endmacro (do_create_variables_for_core_library)

include_directories(${Eigen3_DIR} ${DO_INCLUDE_DIR})

if (DO_USE_FROM_SOURCE)
    do_set_core_source_dir()
    do_list_core_source_files()
    do_create_variables_for_core_library()
    do_append_library(
        Core STATIC
        "${DO_Core_SOURCE_DIR}"
        "${DO_Core_FILES}" ""
        "${DO_Core_LIBRARIES}"
    )
else ()
    do_create_variables_for_core_library()
endif ()