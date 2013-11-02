macro (do_set_featuredetectors_source_dir)
    set(DO_FeatureDetectors_SOURCE_DIR ${DO_SOURCE_DIR}/FeatureDetectors)
endmacro (do_set_featuredetectors_source_dir)

macro (do_list_featuredetectors_source_files)
    # Master header file
    set(DO_FeatureDetectors_MASTER_HEADER ${DO_SOURCE_DIR}/FeatureDetectors.hpp)
    source_group("Master Header File" FILES ${DO_FeatureDetectors_MASTER_HEADER})
    # Header files
    file(GLOB DO_FeatureDetectors_HEADER_FILES
         ${DO_FeatureDetectors_SOURCE_DIR}/*.hpp)
    # Source files
    file(GLOB DO_FeatureDetectors_SOURCE_FILES
         ${DO_FeatureDetectors_SOURCE_DIR}/*.cpp)
    # All header files here
    set(DO_FeatureDetectors_HEADER_FILES
        ${DO_FeatureDetectors_MASTER_HEADER}
        ${DO_FeatureDetectors_HEADER_FILES})
    # Organize source files as follows.
    source_group("Interest Point Detection" FILES
                 ${DO_FeatureDetectors_SOURCE_DIR}/LoG.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/LoG.cpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/DoG.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/DoG.cpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/Harris.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/Harris.cpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/Hessian.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/Hessian.cpp)
    source_group("Extremum Filtering and Refinement" FILES
                 ${DO_FeatureDetectors_SOURCE_DIR}/RefineExtremum.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/RefineExtremum.cpp)
    source_group("Affine Shape Adaptation" FILES
                 ${DO_FeatureDetectors_SOURCE_DIR}/AffineShapeAdaptation.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/AffineShapeAdaptation.cpp)
    source_group("Utilities and Debug" FILES
                 ${DO_FeatureDetectors_SOURCE_DIR}/StdVectorHelpers.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/Subimage.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/Debug.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/Debug.cpp)
    source_group("Adaptive Non Maximal Suppression" FILES
                 ${DO_FeatureDetectors_SOURCE_DIR}/AdaptiveNonMaximalSuppression.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/AdaptiveNonMaximalSuppression.cpp)
endmacro (do_list_featuredetectors_source_files)


macro (do_load_packages_for_featuredetectors_library)
    include(${DO_Core_USE_FILE})
    include(${DO_Graphics_USE_FILE})
    include(${DO_Features_USE_FILE})
    include_directories(${DO_ThirdParty_DIR})
endmacro (do_load_packages_for_featuredetectors_library)

macro (do_create_variables_for_featuredetectors_library)
    set(DO_FeatureDetectors_LIBRARIES DO_FeatureDetectors)
    set(DO_FeatureDetectors_LINK_LIBRARIES
        ${DO_Features_LIBRARIES} ${DO_Graphics_LIBRARIES})
endmacro (do_create_variables_for_featuredetectors_library)

do_load_packages_for_featuredetectors_library()

if (DO_USE_FROM_SOURCE)
    get_property(DO_FeatureDetectors_ADDED GLOBAL PROPERTY _DO_FeatureDetectors_INCLUDED)
    if (NOT DO_FeatureDetectors_ADDED)
        do_set_featuredetectors_source_dir()
        do_list_featuredetectors_source_files()
        do_create_variables_for_featuredetectors_library()
    endif ()
    
    # Static library
    do_append_library(
        FeatureDetectors STATIC
        "${DO_SOURCE_DIR}"
        "${DO_FeatureDetectors_HEADER_FILES}"
        "${DO_FeatureDetectors_SOURCE_FILES}"
        "${DO_FeatureDetectors_LINK_LIBRARIES}"
    )
    do_set_specific_target_properties(DO_FeatureDetectors DO_STATIC)
      
    # Shared library
    if (DO_BUILD_SHARED_LIBS)
      do_append_library(
        FeatureDetectors_SHARED SHARED
        "${DO_SOURCE_DIR}"
        "${DO_FeatureDetectors_HEADER_FILES}"
        "${DO_FeatureDetectors_SOURCE_FILES}"
        "${DO_FeatureDetectors_LINK_LIBRARIES}"
      )
      do_set_specific_target_properties(DO_FeatureDetectors DO_EXPORTS)
    endif ()
endif()
