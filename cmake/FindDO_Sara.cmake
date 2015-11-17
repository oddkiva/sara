if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()


# Load DO-specific macros
include(sara_macros)


# Specify DO-Sara version.
include(sara_version)


# Debug message.
sara_step_message("FindDO_Sara running for project '${PROJECT_NAME}'")


# Setup DO-CV once for all for every test projects in the 'test' directory.
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
  sara_append_components(DO_Sara_COMPONENTS Core)
  sara_append_components(DO_Sara_COMPONENTS ImageIO)
  sara_append_components(DO_Sara_COMPONENTS VideoIO)
  sara_append_components(DO_Sara_COMPONENTS Graphics)

  # KDTree for fast neighbor search.
  sara_append_components(DO_Sara_COMPONENTS KDTree)
  # Image processing
  sara_append_components(DO_Sara_COMPONENTS ImageProcessing)
  # Geometry
  sara_append_components(DO_Sara_COMPONENTS Geometry)
  # Feature detection and description
  sara_append_components(DO_Sara_COMPONENTS Features)
  sara_append_components(DO_Sara_COMPONENTS FeatureDetectors)
  sara_append_components(DO_Sara_COMPONENTS FeatureDescriptors)
  # Feature matching
  sara_append_components(DO_Sara_COMPONENTS Match)
  sara_append_components(DO_Sara_COMPONENTS FeatureMatching)

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


# List the compile flags needed by DO-CV.
set(SARA_DEFINITIONS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}")
if (SARA_USE_STATIC_LIBS OR NOT SARA_BUILD_SHARED_LIBS)
  add_definitions("-DDO_SARA_STATIC")
endif ()


# Include directories.
find_path(
  DO_Sara_INCLUDE_DIRS
  NAMES DO/Sara/Core.hpp DO/Sara/Defines.hpp DO/Sara/Graphics.hpp
  PATHS /usr/include
        /usr/local/include
        "C:/Program Files/DO-Sara/include")


# 'find_package(DO_Sara COMPONENTS Core Graphics ... REQUIRED)' is called.
if (NOT DO_Sara_FIND_COMPONENTS)
  set(DO_Sara_USE_COMPONENTS ${DO_Sara_COMPONENTS})
else ()
  # Verbose comment.
  sara_step_message("Requested libraries by project '${PROJECT_NAME}':")
  foreach (component ${DO_Sara_FIND_COMPONENTS})
    sara_substep_message ("- ${component}")
  endforeach (component)

  # Check that all the components exist.
  set(DO_Sara_USE_COMPONENTS "")
  foreach (component ${DO_Sara_FIND_COMPONENTS})
    list(FIND DO_Sara_COMPONENTS ${component} COMPONENT_INDEX)
    if (COMPONENT_INDEX EQUAL -1)
      message (FATAL_ERROR "[Sara] ${component} does not exist!")
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

  if (SARA_USE_FROM_SOURCE)
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

      if (NOT COMPONENT STREQUAL "ImageProcessing")
        find_path(DO_Sara_${COMPONENT}_INCLUDE_DIR
          NAMES ${COMPONENT}.hpp
          PATHS /usr/include /usr/local/include /opt/local/include
                "C:/Program Files/DO-Sara/include"
          PATH_SUFFIXES DO/Sara)

        if (DO_USE_STATIC_LIBS)
          set (_library_name "DO_Sara_${COMPONENT}-s")
          set (_library_name_debug "DO_Sara_${COMPONENT}-sd")
        else ()
          set (_library_name "DO_Sara_${COMPONENT}")
          set (_library_name_debug "DO_Sara_${COMPONENT}-d")
        endif ()

        find_library(DO_Sara_${COMPONENT}_DEBUG_LIBRARIES
          NAMES ${_library_name_debug}
          PATHS /usr/lib /usr/local/lib /opt/local/lib
                "C:/Program Files/DO-Sara-Debug/lib"
          PATH_SUFFIXES DO/Sara)

        find_library(DO_Sara_${COMPONENT}_RELEASE_LIBRARIES
          NAMES ${_library_name}
          PATHS /usr/lib /usr/local/lib /opt/local/lib
                "C:/Program Files/DO-Sara/lib"
          PATH_SUFFIXES DO/Sara)

        if (NOT DO_USE_STATIC_LIBS AND NOT DO_Sara_${COMPONENT}_DEBUG_LIBRARIES)
          set(
            DO_Sara_${COMPONENT}_LIBRARIES
            ${DO_Sara_${COMPONENT}_RELEASE_LIBRARIES}
            CACHE STRING "")
        else ()
          set(DO_Sara_${COMPONENT}_LIBRARIES
            debug ${DO_Sara_${COMPONENT}_DEBUG_LIBRARIES}
            optimized ${DO_Sara_${COMPONENT}_RELEASE_LIBRARIES}
            CACHE STRING "")
        endif ()

        if (DO_USE_STATIC_LIBS)
          if (NOT DO_Sara_${COMPONENT}_DEBUG_LIBRARIES OR
              NOT DO_Sara_${COMPONENT}_RELEASE_LIBRARIES)
            message(FATAL_ERROR "DO_Sara_${COMPONENT} is missing!")
          endif ()
        elseif (NOT DO_Sara_${COMPONENT}_RELEASE_LIBRARIES)
          message(FATAL_ERROR "DO_Sara_${COMPONENT} is missing!")
        endif ()

        if (DO_Sara_${COMPONENT}_LIBRARIES)
          list(APPEND DO_Sara_LIBRARIES ${DO_Sara_${COMPONENT}_LIBRARIES})
        endif ()

      endif ()

      if ("${COMPONENT}" STREQUAL "Graphics")
        list(APPEND DO_Sara_LIBRARIES
          Qt5::OpenGL Qt5::Widgets ${OPENGL_LIBRARIES})
      endif ()

      if ("${COMPONENT}" STREQUAL "ImageIO")
        # EXIF library
        find_package(EasyEXIF)

        # JPEG, PNG, TIFF...
        set(IMAGE_IO_LIBRARIES JPEG PNG TIFF ZLIB)
        foreach (IMAGE_IO_LIB ${IMAGE_IO_LIBRARIES})
          if (WIN32)
            find_library(${IMAGE_IO_LIB}_DEBUG_LIBRARY
              NAMES ${IMAGE_IO_LIB}-d
              PATHS "C:/Program Files/DO-Sara-Debug/lib")
            find_library(${IMAGE_IO_LIB}_RELEASE_LIBRARY
              NAMES ${IMAGE_IO_LIB}
              PATHS "C:/Program Files/DO-Sara/lib")
            if (NOT DO_USE_STATIC_LIBS AND NOT ${IMAGE_IO_LIB}_DEBUG_LIBRARY)
              set(${IMAGE_IO_LIB}_LIBRARY ${${IMAGE_IO_LIB}_RELEASE_LIBRARY})
            else ()
              set(${IMAGE_IO_LIB}_LIBRARY
                debug ${${IMAGE_IO_LIB}_DEBUG_LIBRARY}
                optimized ${${IMAGE_IO_LIB}_RELEASE_LIBRARY})
            endif ()
          else ()
            find_package(JPEG REQUIRED)
            find_package(PNG REQUIRED)
            find_package(TIFF REQUIRED)
            find_package(ZLIB REQUIRED)
          endif ()
        endforeach()

        # Add these image I/O libraries to the dependencies.
        list(APPEND
          DO_Sara_LIBRARIES
          ${EasyEXIF_LIBRARIES}
          ${JPEG_LIBRARY} ${PNG_LIBRARY} ${TIFF_LIBRARY}
          ${ZLIB_LIBRARY}
        )
      endif ()

    endforeach()
    message("DO_Sara_LIBRARIES = ${DO_Sara_LIBRARIES}")
  endif ()

endif ()
