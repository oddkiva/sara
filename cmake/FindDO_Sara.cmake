# Load DO-specific macros
include(DOMacros)


# Specify DO-Sara version.
include(DO_Sara_version)


# Debug message.
do_step_message("FindDO running for project '${PROJECT_NAME}'")


# Setup DO++ once for all for every test projects in the 'test' directory.
if (NOT DO_Sara_FOUND)

  # Do we build from source?
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindDO_Sara.cmake")
    # DO++ needs to be built or used from source
    message(STATUS "Building DO++ from source")
    set(DO_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(DO_INCLUDE_DIR ${DO_DIR}/src)

  elseif (UNIX AND EXISTS "/usr/share/DO/Sara/cmake/FindDO_Sara.cmake")
    do_dissect_version()
    do_get_os_info()

  else ()
    message(FATAL_ERROR "DO-Sara is not found!")
  endif ()

  # DEBUG
  do_step_message("Found DO-Sara libraries in directory:")
  message(STATUS "  - DO_DIR = '${DO_DIR}'")

  # Set third-party software directories
  set(DO_SOURCE_DIR ${DO_DIR}/src/DO/Sara)
  set(DO_ThirdParty_DIR ${DO_DIR}/third-party)

  # List the available component libraries in DO++
  # Foundational libraries
  do_append_components(DO_COMPONENTS Core)
  do_append_components(DO_COMPONENTS ImageIO)
  do_append_components(DO_COMPONENTS VideoIO)
  do_append_components(DO_COMPONENTS Graphics)

  # KDTree for fast neighbor search.
  do_append_components(DO_COMPONENTS KDTree)
  # Image processing
  do_append_components(DO_COMPONENTS ImageProcessing)
  # Geometry
  do_append_components(DO_COMPONENTS Geometry)
  # Feature detection and description
  do_append_components(DO_COMPONENTS Features)
  #do_append_components(DO_COMPONENTS FeatureDetectors)
  #do_append_components(DO_COMPONENTS FeatureDescriptors)
  # Feature matching
  #do_append_components(DO_COMPONENTS Match)
  #do_append_components(DO_COMPONENTS FeatureMatching)

  # DEBUG: Print the list of component libraries
  do_step_message("Currently available component libraries:")
  foreach (component ${DO_COMPONENTS})
    message (STATUS "  - ${component}")
  endforeach (component)

  # Configure compiler for the specific project.
  include (DOConfigureCompiler)

  # Set DO_Sara as found.
  set(DO_Sara_FOUND TRUE)

endif ()


# Check that the requested libraries exists when, e.g.:
# 'find_package(DO_Sara COMPONENTS Core Graphics ... REQUIRED)' is called.
if (DO_Sara_FIND_COMPONENTS)
  message("HELLOOO")
  # Configure compiler for the specific project.
  include (DOConfigureCompiler)

  # Verbose comment.
  do_step_message("Requested libraries by project '${PROJECT_NAME}':")
  foreach (component ${DO_Sara_FIND_COMPONENTS})
    do_substep_message ("- ${component}")
  endforeach (component)

  # Check that all the components exist.
  set(DO_USE_COMPONENTS "")
  foreach (component ${DO_Sara_FIND_COMPONENTS})
    list(FIND DO_COMPONENTS ${component} COMPONENT_INDEX)
    if (COMPONENT_INDEX EQUAL -1)
      message (FATAL_ERROR "[DO] ${component} does not exist!")
    else ()
      set(DO_${component}_FOUND TRUE)
      list (APPEND DO_USE_COMPONENTS ${component})
    endif ()
  endforeach (component)

  if (POLICY CMP0011)
    cmake_policy(SET CMP0011 OLD)
  endif (POLICY CMP0011)

  # Retrieve the set of dependencies when linking projects with DO-CV.
  set(DO_LIBRARIES "")
  foreach (COMPONENT ${DO_USE_COMPONENTS})
    include(UseDO${COMPONENT})
    if ("${DO_LIBRARIES}" STREQUAL "" AND
        NOT "${DO_${COMPONENT}_LIBRARIES}" STREQUAL "")
      set (DO_LIBRARIES "${DO_${COMPONENT}_LIBRARIES}")
    elseif (NOT "${DO_${COMPONENT}_LIBRARIES}" STREQUAL "")
      set(DO_LIBRARIES "${DO_LIBRARIES};${DO_${COMPONENT}_LIBRARIES}")
    endif ()
  endforeach ()
endif ()


# List the compile flags needed by DO-CV.
set(DO_DEFINITIONS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}")
if (DO_USE_FROM_SOURCE)
  set(DO_DEFINITIONS "${DO_DEFINITIONS} -DDO_STATIC")
endif ()
