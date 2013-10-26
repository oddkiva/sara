macro (do_set_featuredetectorwrappers_source_dir)
    set(DO_FeatureDetectorWrappers_SOURCE_DIR ${DO_SOURCE_DIR}/FeatureDetectorWrappers)
endmacro (do_set_featuredetectorwrappers_source_dir)

macro (do_list_featuredetectorwrappers_source_files)
    # Master header file
    set(DO_FeatureDetectorWrappers_MASTER_HEADER ${DO_SOURCE_DIR}/FeatureDetectorWrappers.hpp)
    source_group("Master Header File" FILES ${DO_FeatureDetectorWrappers_MASTER_HEADER})
    # Header files
    file(GLOB DO_FeatureDetectorWrappers_HEADER_FILES
         ${DO_FeatureDetectorWrappers_SOURCE_DIR}/*.hpp)
    # Source files
    file(GLOB DO_FeatureDetectorWrappers_SOURCE_FILES
         ${DO_FeatureDetectorWrappers_SOURCE_DIR}/*.cpp)
    # All header files here
    set(DO_FeatureDetectorWrappers_HEADER_FILES
        ${DO_FeatureDetectorWrappers_MASTER_HEADER}
        ${DO_FeatureDetectorWrappers_HEADER_FILES})
    source_group("Keypoint Detector Wrapper (Lowe's SIFT implementation)" FILES
                 ${DO_FeatureDetectors_SOURCE_DIR}/DoGSiftDetector.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/DoGSiftDetector.cpp)
    source_group("Keypoint Detector Wrapper (Mikolajczyk's binary)" FILES
                 ${DO_FeatureDetectors_SOURCE_DIR}/HarAffSiftDetector.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/HarAffSiftDetector.cpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/HesAffSiftDetector.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/HesAffSiftDetector.cpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/MserSiftDetector.hpp
                 ${DO_FeatureDetectors_SOURCE_DIR}/MserSiftDetector.cpp)

endmacro (do_list_featuredetectorwrappers_source_files)


macro (do_load_packages_for_featuredetectorwrappers_library)
    include(${DO_Core_USE_FILE})
    include(${DO_Graphics_USE_FILE})
    include(${DO_Features_USE_FILE})
    include_directories(${DO_ThirdParty_DIR})
endmacro (do_load_packages_for_featuredetectorwrappers_library)

macro (do_create_variables_for_featuredetectorwrappers_library)
    set(DO_FeatureDetectorWrappers_LIBRARIES DO_FeatureDetectorWrappers)
    set(DO_FeatureDetectorWrappers_LINK_LIBRARIES
        ${DO_Features_LIBRARIES} ${DO_Graphics_LIBRARIES})
endmacro (do_create_variables_for_featuredetectorwrappers_library)

do_load_packages_for_featuredetectorwrappers_library()

if (DO_USE_FROM_SOURCE)
    get_property(DO_FeatureDetectorWrappers_ADDED GLOBAL PROPERTY _DO_FeatureDetectorWrappers_INCLUDED)
    if (NOT DO_FeatureDetectorWrappers_ADDED)
        do_set_featuredetectorwrappers_source_dir()
        do_list_featuredetectorwrappers_source_files()
        do_create_variables_for_featuredetectorwrappers_library()
    endif ()
    
    # Static library
    do_append_library(
        FeatureDetectorWrappers STATIC
        "${DO_SOURCE_DIR}"
        "${DO_FeatureDetectorWrappers_HEADER_FILES}"
        "${DO_FeatureDetectorWrappers_SOURCE_FILES}"
        "${DO_FeatureDetectorWrappers_LINK_LIBRARIES}"
    )
    do_set_specific_target_properties(DO_FeatureDetectorWrappers DO_STATIC)
    # Specify the location of external binaries.
    set_target_properties(
      DO_FeatureDetectorWrappers PROPERTIES
      COMPILE_FLAGS -DEXTERNBINDIR="\"${DO_ThirdParty_DIR}/Mikolajczyk\""
    )
      
    # Shared library
    if (DO_BUILD_SHARED_LIBS)
      do_append_library(
        FeatureDetectorWrappers_SHARED SHARED
        "${DO_SOURCE_DIR}"
        "${DO_FeatureDetectorWrappers_HEADER_FILES}"
        "${DO_FeatureDetectorWrappers_SOURCE_FILES}"
        "${DO_FeatureDetectorWrappers_LINK_LIBRARIES}"
      )
      do_set_specific_target_properties(DO_FeatureDetectorWrappers DO_EXPORTS)
      # Specify the location of external binaries.
      set_target_properties(
        DO_FeatureDetectorWrappers_SHARED PROPERTIES
        COMPILE_FLAGS -DEXTERNBINDIR="\"${DO_ThirdParty_DIR}/Mikolajczyk\""
      )
    endif ()
endif()
