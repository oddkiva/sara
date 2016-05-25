if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

# Load DO-specific macros
include(sara_macros)

# Specify DO-Sara version.
include(sara_version)

# Debug message.
sara_step_message("FindDO_Sara running for project '${PROJECT_NAME}'")



# ============================================================================ #
# Setup Sara once for all for every test projects in the 'test' directory.
#
if (NOT DO_Sara_FOUND)

  # Convenience variables used later in 'UseDOSaraXXX.cmake' scripts.
  set(DO_Sara_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "")
  set(DO_Sara_INCLUDE_DIR ${DO_Sara_DIR}/src CACHE STRING "")
  set(DO_Sara_SOURCE_DIR ${DO_Sara_DIR}/src/DO/Sara)
  set(DO_Sara_ThirdParty_DIR ${DO_Sara_DIR}/third-party CACHE STRING "")

  message("DO_Sara_SOURCE_DIR = ${DO_Sara_SOURCE_DIR}")

  # List the available component libraries in Sara libraries.

  # Foundational libraries.
  sara_append_components(DO_Sara_COMPONENTS Core)
  sara_append_components(DO_Sara_COMPONENTS Graphics)

  # Image and Video I/O.
  sara_append_components(DO_Sara_COMPONENTS ImageIO)
  if (SARA_BUILD_VIDEOIO)
    sara_append_components(DO_Sara_COMPONENTS VideoIO)
  endif ()

  # Image processing.
  sara_append_components(DO_Sara_COMPONENTS ImageProcessing)

  # Feature detection and description.
  sara_append_components(DO_Sara_COMPONENTS Features)
  sara_append_components(DO_Sara_COMPONENTS FeatureDetectors)
  sara_append_components(DO_Sara_COMPONENTS FeatureDescriptors)

  # Feature matching.
  sara_append_components(DO_Sara_COMPONENTS Match)
  sara_append_components(DO_Sara_COMPONENTS FeatureMatching)

  # Disjoint sets.
  sara_append_components(DO_Sara_COMPONENTS DisjointSets)

  # Geometry.
  sara_append_components(DO_Sara_COMPONENTS Geometry)

  # KDTree for fast neighbor search.
  sara_append_components(DO_Sara_COMPONENTS KDTree)

  # DEBUG: Print the list of component libraries
  sara_step_message("Currently available components in DO-Sara:")
  foreach (component ${DO_Sara_COMPONENTS})
    message (STATUS "  - ${component}")
  endforeach (component)

  # Set DO_Sara as found.
  set(DO_Sara_FOUND TRUE)

endif ()


# Configure compiler for the specific project.
include (sara_configure_cxx_compiler)


# List the compile flags needed by Sara.
set(SARA_DEFINITIONS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}")
if (SARA_USE_STATIC_LIBS OR NOT SARA_BUILD_SHARED_LIBS)
  add_definitions("-DDO_SARA_STATIC")
endif ()



# ============================================================================ #
# 'find_package(DO_Sara COMPONENTS Core Graphics... [REQUIRED|QUIET])'
#
if (NOT DO_Sara_FIND_COMPONENTS)
  set(DO_Sara_USE_COMPONENTS ${DO_Sara_COMPONENTS})
else ()

  # Verbose comment.
  sara_step_message("Requested libraries by project '${PROJECT_NAME}':")
  foreach (component ${DO_Sara_FIND_COMPONENTS})
    sara_substep_message ("- ${component}")
  endforeach (component)

  # Check that all components exist.
  set(DO_Sara_USE_COMPONENTS "")
  foreach (component ${DO_Sara_FIND_COMPONENTS})

    # By default, mark the requested component as not found.
    set(DO_Sara_${component}_FOUND FALSE)

    # Now check if the requested component exists.
    list(FIND DO_Sara_COMPONENTS ${component} COMPONENT_INDEX)
    if (NOT COMPONENT_INDEX EQUAL -1)
      set(DO_Sara_${component}_FOUND TRUE)
      list (APPEND DO_Sara_USE_COMPONENTS ${component})
    endif ()

    # Stop if REQUIRED option was given.
    if (NOT DO_Sara_${component}_FOUND AND DO_Sara_FIND_REQUIRED)
      message (FATAL_ERROR "[Sara] ${component} does not exist!")
    endif ()

  endforeach (component)

  if (POLICY CMP0011)
    cmake_policy(SET CMP0011 OLD)
  endif (POLICY CMP0011)


  # Retrieve the set of dependencies when linking projects with Sara.
  set(DO_Sara_LIBRARIES "")

  foreach (COMPONENT ${DO_Sara_USE_COMPONENTS})
    include(UseDOSara${COMPONENT})

    if ("${DO_Sara_LIBRARIES}" STREQUAL "" AND
        NOT "${DO_Sara_${COMPONENT}_LIBRARIES}" STREQUAL "")
      set (DO_Sara_LIBRARIES "${DO_Sara_${COMPONENT}_LIBRARIES}")
    elseif (NOT "${DO_Sara_${COMPONENT}_LIBRARIES}" STREQUAL "")
      set(DO_Sara_LIBRARIES
        "${DO_Sara_LIBRARIES};${DO_Sara_${COMPONENT}_LIBRARIES}")
    endif ()
  endforeach ()

endif ()
