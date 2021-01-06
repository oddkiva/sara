# ============================================================================ #
# Setup Sara once for all for every test and example projects.
#
if (NOT DO_Sara_FOUND)
  # Convenience variables used later in 'UseDOSaraXXX.cmake' scripts.
  set(DO_Sara_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "")
  set(DO_Sara_INCLUDE_DIR ${DO_Sara_DIR}/cpp/src CACHE STRING "")
  set(DO_Sara_SOURCE_DIR ${DO_Sara_DIR}/cpp/src/DO/Sara)
  set(DO_Sara_ThirdParty_DIR ${DO_Sara_DIR}/cpp/third-party CACHE STRING "")
  sara_step_message("DO_Sara_SOURCE_DIR = ${DO_Sara_SOURCE_DIR}")

  # Load Sara specific macros
  include(sara_macros)

  # Specify DO-Sara version.
  include(sara_version)

  # Configure compiler for the specific project.
  include(sara_configure_cxx_compiler)

  # List the available components.
  sara_populate_available_components()

  # Automatically link Qt executables to qtmain target on Windows.
  if(POLICY CMP0020)
    cmake_policy(SET CMP0020 NEW)
  endif()

  # Mark Sara as found.
  set(DO_Sara_FOUND TRUE)
endif ()


# ============================================================================ #
# 'find_package(DO_Sara COMPONENTS Core Graphics... [REQUIRED|QUIET])'
#
# Debug message.
sara_step_message("FindDO_Sara running for project '${PROJECT_NAME}'")

# Set the compile flags needed by Sara.
set(SARA_DEFINITIONS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}")
if (SARA_USE_STATIC_LIBS OR NOT SARA_BUILD_SHARED_LIBS)
  add_definitions("-DDO_SARA_STATIC")
endif ()

if (NOT DO_Sara_FIND_COMPONENTS)
  set(DO_Sara_USE_COMPONENTS ${DO_Sara_COMPONENTS})
else ()
  # Check the requested components exist.
  sara_check_requested_components()

  # Reset the list of component libraries.
  set(DO_Sara_LIBRARIES "")

  # Populate the list of components libraries which we need to link with.
  foreach (COMPONENT ${DO_Sara_USE_COMPONENTS})
    sara_message("Creating library DO_Sara_${COMPONENT}...")
    include(${DO_Sara_SOURCE_DIR}/UseDOSara${COMPONENT}.cmake)

    if ("${DO_Sara_LIBRARIES}" STREQUAL "" AND
        NOT "${DO_Sara_${COMPONENT}_LIBRARIES}" STREQUAL "")
      set (DO_Sara_LIBRARIES "${DO_Sara_${COMPONENT}_LIBRARIES}")
    elseif (NOT "${DO_Sara_${COMPONENT}_LIBRARIES}" STREQUAL "")
      set(DO_Sara_LIBRARIES
        "${DO_Sara_LIBRARIES};${DO_Sara_${COMPONENT}_LIBRARIES}")
    endif ()
  endforeach ()
endif ()

# Debug message.
sara_step_message("DO_Sara_LIBRARIES = ${DO_Sara_LIBRARIES}")
