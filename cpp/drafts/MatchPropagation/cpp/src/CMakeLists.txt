find_package(DO_Sara COMPONENTS
  Features FeatureMatching FileSystem Graphics Geometry KDTree Match REQUIRED)

include (CMakePackageConfigHelpers)
include (CMakeFindDependencyMacro)


macro (sara_make_library _library_name
                         _include_dirs
                         _hdr_files
                         _src_files
                         _lib_dependencies)
  # 1. Verbose comment.
  sara_message("Creating project 'DO::Sara::${_library_name}'")

  # 2. Bookmark the project to make sure the library is created only once.
  set_property(GLOBAL PROPERTY _DO-Sara-${_library_name}-INCLUDED 1)

  sara_step_message("[${_library_name}] include = '${_include_dirs}'")

  # 3. Create the project:
  if (NOT "${_src_files}" STREQUAL "")
    # - Case 1: the project contains 'cpp' source files
    #   Specify the source files.
    add_library(DO-Sara-${_library_name} ${_hdr_files} ${_src_files})

    # Create a namespaced alias for future library reuse.
    add_library(DO::Sara::${_library_name} ALIAS DO-Sara-${_library_name})

    sara_step_message("[${_library_name}] headers = '${_hdr_files}'")
    sara_step_message("[${_library_name}] sources = '${_src_files}'")
    sara_step_message("[${_library_name}] deps    = '${_lib_dependencies}'")

    # 4. Include third-party library directories.
    if (NOT "${_include_dirs}" STREQUAL "")
      target_include_directories(DO-Sara-${_library_name}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
               $<INSTALL_INTERFACE:include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    endif ()

    # 5. Link with other libraries.
    sara_step_message("Linking project 'DO::Sara::${_library_name}' with "
                      "'${_lib_dependencies}'")

    target_link_libraries(DO-Sara-${_library_name} PUBLIC ${_lib_dependencies})

    # Form the compiled library output name.
    set(_library_output_basename DO-Sara-${_library_name})
    if (SARA_BUILD_SHARED_LIBS)
      set (_library_output_name "${_library_output_basename}")
      set (_library_output_name_debug "${_library_output_basename}-d")
    else ()
      set (_library_output_name "${_library_output_basename}-s")
      set (_library_output_name_debug "${_library_output_basename}-sd")
    endif ()

    # Specify output name and version.
    set_target_properties(DO-Sara-${_library_name}
      PROPERTIES VERSION ${DO_Sara_VERSION}
                 SOVERSION ${DO_Sara_SOVERSION}
                 OUTPUT_NAME ${_library_output_name}
                 OUTPUT_NAME_DEBUG ${_library_output_name_debug})

    # Set correct compile definitions when building the libraries.
    if (SARA_BUILD_SHARED_LIBS)
      set(_library_defs "DO_SARA_EXPORTS")
    else ()
      set(_library_defs "DO_SARA_STATIC")
    endif ()
    set_target_properties(DO-Sara-${_library_name}
      PROPERTIES COMPILE_DEFINITIONS ${_library_defs})
  else ()

    # - Case 2: the project is a header-only library
    #   Specify the source files.
    message(STATUS
      "[Sara] No linking needed for header-only project "
      "'DO-Sara-${_library_name}'")
    add_custom_target(DO_Sara_${_library_name} SOURCES ${_hdr_files})
  endif ()

  # 5. Put the library into the folder "DO Libraries".
  set_property(
    TARGET DO-Sara-${_library_name} PROPERTY
    FOLDER "DO Sara Libraries")


  # Figure out the rest later.
  # 6. Specify where to install the library.
  # install(TARGETS DO-Sara-${_library_name}
  #   EXPORT DO-Sara-${_library_name}-Targets
  #   RUNTIME  DESTINATION bin     COMPONENT Libraries
  #   LIBRARY  DESTINATION lib     COMPONENT Libraries
  #   ARCHIVE  DESTINATION lib     COMPONENT Libraries
  #   INCLUDES DESTINATION include COMPONENT LIBRARIES)

  # # 7. Versioning.
  # write_basic_package_version_file(
  #   "DO-Sara-${_library_name}-Version.cmake"
  #   VERSION DO-Sara-${DO_Sara_VERSION}
  #   COMPATIBILITY SameMajorVersion)

  # install(FILES "DO-Sara-${_library_name}-Config.cmake"
  #               "DO-Sara-${_library_name}-Version.cmake"
  #         DESTINATION lib/cmake/DO)

  # 8. For library reuse.
  # install(EXPORT DO-Sara-${_library_name}-Targets
  #   FILE DO-Sara-${_library_name}-Targets.cmake
  #   NAMESPACE DO::Sara::
  #   DESTINATION lib/cmake/DO)
  # foreach (lib ${_lib_dependencies})
  #   find_dependency(${_lib} ${DO_Sara_VERSION})
  # endforeach ()
  # include("${CMAKE_CURRENT_LIST_DIR}/DO-Sara-${_library_name}-Targets.cmake")

endmacro ()


# ==============================================================================
# Region data structures
set(Region_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(Region_INC Region.hpp RegionBoundary.hpp)
set(Region_SRC Region.cpp RegionBoundary.cpp)
set(Region_DEPS "${DO_Sara_LIBRARIES}")
message("Region_DEPS = ${Region_DEPS}")

sara_make_library("Region"
  "${Region_INC_DIR}"
  "${Region_INC}"
  "${Region_SRC}"
  "${Region_DEPS}")


# ==============================================================================
# The region growing algorithm.
set(MatchPropagation_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MatchPropagation_INC
  LocalAffineConsistency.hpp
  MatchNeighborhood.hpp
  GrowthParams.hpp
  DynamicMatchGraph.hpp
  GrowRegionDebugMacros.hpp
  GrowRegion.hpp
  GrowMultipleRegions.hpp
  RegionGrowingAnalyzer.hpp
  Statistics.hpp)
set(MatchPropagation_SRC
  LocalAffineConsistency.cpp
  MatchNeighborhood.cpp
  GrowRegion.cpp
  GrowMultipleRegions.cpp
  RegionGrowingAnalyzer.cpp
  Statistics.cpp)
set(MatchPropagation_DEPS "${DO_Sara_LIBRARIES};DO::Sara::Region")

sara_make_library("MatchPropagation"
  "${MatchPropagation_INC_DIR}"
  "${MatchPropagation_INC}"
  "${MatchPropagation_SRC}"
  "${MatchPropagation_DEPS}")

add_subdirectory(EmpiricalValidation)
#add_subdirectory(HyperParameterLearning)

