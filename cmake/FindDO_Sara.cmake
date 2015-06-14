# Load DO-specific macros
include(do_macros)


# Specify DO-Sara version.
include(DO_Sara_version)


# Debug message.
do_step_message("FindDO_Sara running for project '${PROJECT_NAME}'")


# Setup DO++ once for all for every test projects in the 'test' directory.
if (NOT DO_Sara_FOUND)

  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindDO_Sara.cmake")
    message(STATUS "Building DO-Sara from source")

    # Convenience variables used later in 'UseDOSaraXXX.cmake' scripts.
    set(DO_Sara_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "")
    set(DO_Sara_INCLUDE_DIR ${DO_Sara_DIR}/src CACHE STRING "")
    set(DO_Sara_SOURCE_DIR ${DO_Sara_DIR}/src/DO/Sara)
    set(DO_Sara_ThirdParty_DIR ${DO_Sara_DIR}/third-party CACHE STRING "")

    message("DO_Sara_SOURCE_DIR = ${DO_Sara_SOURCE_DIR}")

  endif ()

  # List the available component libraries in DO++
  # Foundational libraries
  do_append_components(DO_Sara_COMPONENTS Core)
  do_append_components(DO_Sara_COMPONENTS ImageIO)
  do_append_components(DO_Sara_COMPONENTS VideoIO)
  do_append_components(DO_Sara_COMPONENTS Graphics)

  # KDTree for fast neighbor search.
  do_append_components(DO_Sara_COMPONENTS KDTree)
  # Image processing
  do_append_components(DO_Sara_COMPONENTS ImageProcessing)
  # Geometry
  do_append_components(DO_Sara_COMPONENTS Geometry)
  # Feature detection and description
  do_append_components(DO_Sara_COMPONENTS Features)
  #do_append_components(DO_Sara_COMPONENTS FeatureDetectors)
  #do_append_components(DO_Sara_COMPONENTS FeatureDescriptors)
  # Feature matching
  #do_append_components(DO_Sara_COMPONENTS Match)
  #do_append_components(DO_Sara_COMPONENTS FeatureMatching)

  # DEBUG: Print the list of component libraries
  do_step_message("Currently available components in DO-Sara:")
  foreach (component ${DO_Sara_COMPONENTS})
    message (STATUS "  - ${component}")
  endforeach (component)

  # Set DO_Sara as found.
  set(DO_Sara_FOUND TRUE)

endif ()


# Configure compiler for the specific project.
include (do_configure_cxx_compiler)


# List the compile flags needed by DO-CV.
set(DO_DEFINITIONS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}")
if (DO_USE_FROM_SOURCE)
  set(DO_DEFINITIONS "${DO_DEFINITIONS} -DDO_STATIC")
endif ()


# 'find_package(DO_Sara COMPONENTS Core Graphics ... REQUIRED)' is called.
if (DO_Sara_FIND_COMPONENTS)
  # Verbose comment.
  do_step_message("Requested libraries by project '${PROJECT_NAME}':")
  foreach (component ${DO_Sara_FIND_COMPONENTS})
    do_substep_message ("- ${component}")
  endforeach (component)

  # Check that all the components exist.
  set(DO_Sara_USE_COMPONENTS "")
  foreach (component ${DO_Sara_FIND_COMPONENTS})
    list(FIND DO_Sara_COMPONENTS ${component} COMPONENT_INDEX)
    if (COMPONENT_INDEX EQUAL -1)
      message (FATAL_ERROR "[DO] ${component} does not exist!")
    else ()
      set(DO_Sara_${component}_FOUND TRUE)
      list (APPEND DO_Sara_USE_COMPONENTS ${component})
    endif ()
  endforeach (component)

  if (POLICY CMP0011)
    cmake_policy(SET CMP0011 OLD)
  endif (POLICY CMP0011)


  # Retrieve the set of dependencies when linking projects with DO-CV.
  set(DO_Sara_LIBRARIES "")

  if (DO_USE_FROM_SOURCE)
    foreach (COMPONENT ${DO_Sara_USE_COMPONENTS})
      include(UseDOSara${COMPONENT})

      if ("${DO_Sara_LIBRARIES}" STREQUAL "" AND
          NOT "${DO_Sara_${COMPONENT}_LIBRARIES}" STREQUAL "")
        set (DO_Sara_LIBRARIES "${DO_Sara_${COMPONENT}_LIBRARIES}")
      elseif (NOT "${DO_Sara_${COMPONENT}_LIBRARIES}" STREQUAL "")
        set(DO_Sara_LIBRARIES "${DO_Sara_LIBRARIES};${DO_Sara_${COMPONENT}_LIBRARIES}")
      endif ()
    endforeach ()
  else ()
    foreach (COMPONENT ${DO_Sara_USE_COMPONENTS})
      include(UseDOSara${COMPONENT})

      find_path(DO_Sara_${COMPONENT}_INCLUDE_DIR
        NAMES ${COMPONENT}.hpp
        PATHS /usr/include /usr/local/include /opt/local/include
        PATH_SUFFIXES DO/Sara)

      find_library(DO_Sara_${COMPONENT}_LIBRARIES
        NAMES DO_Sara_${COMPONENT}-${DO_Sara_VERSION}
        PATHS /usr/lib /usr/local/lib /opt/local/lib
        PATH_SUFFIXES DO/Sara)

      if (DO_Sara_${COMPONENT}_LIBRARIES)
        list(APPEND DO_Sara_LIBRARIES ${DO_Sara_${COMPONENT}_LIBRARIES})
      endif ()

      if ("${COMPONENT}" STREQUAL "Graphics")
        list(APPEND DO_Sara_LIBRARIES
          Qt5::OpenGL Qt5::Widgets ${OPENGL_LIBRARIES})
      endif ()
    endforeach()
    message("DO_Sara_LIBRARIES = ${DO_Sara_LIBRARIES}")
  endif ()

endif ()
