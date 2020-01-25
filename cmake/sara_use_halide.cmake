# If the link fails, run `install_name_tool` to update the install name in
# `libHalide.dylib` file.
#
# I have installed halide in `/opt/halide`:
#
# $ cd /opt/halide/bin
# $ install_name_tool -id "/opt/halide/bin/libHalide.dylib" libHalide.dylib
#
# References:
# - https://github.com/halide/Halide/issues/2821
# - https://stackoverflow.com/questions/33991581/install-name-tool-to-update-a-executable-to-search-for-dylib-in-mac-os-x
#
# Then:
# - MacOSX will complain about the unverified dylib file. Click on the help
# button from the dialog box that popped up.
# - Follow the instructions in the user guide to allow the use of this file.

if (NOT HALIDE_DISTRIB_DIR)
  message(FATAL_ERROR "Please specify HALIDE_DISTRIB_DIR to use Halide")
endif ()

find_file(HALIDE_CMAKE_CONFIG_FILEPATH NAMES halide_config.cmake
  PATHS ${HALIDE_DISTRIB_DIR})
find_file(HALIDE_CMAKE_FILEPATH NAMES halide.cmake
  PATHS ${HALIDE_DISTRIB_DIR})

find_path(HALIDE_INCLUDE_DIR NAMES Halide.h
  PATHS ${HALIDE_DISTRIB_DIR}
  PATH_SUFFIXES include)

if (WIN32)
  find_library(HALIDE_LIBRARY_DEBUG NAMES Halide
    PATHS ${HALIDE_DISTRIB_DIR}/Debug)
  find_library(HALIDE_LIBRARY_RELEASE NAMES Halide
    PATHS ${HALIDE_DISTRIB_DIR}/Release)

  set(HALIDE_LIBRARIES
    debug ${HALIDE_LIBRARY_DEBUG}
    optimized ${HALIDE_LIBRARY_RELEASE}
    CACHE STRING "Halide libraries")

  find_file(HALIDE_DLL_DEBUG NAMES Halide.dll
    PATHS ${HALIDE_DISTRIB_DIR}/Debug)
  find_file(HALIDE_DLL_RELEASE NAMES Halide.dll
    PATHS ${HALIDE_DISTRIB_DIR}/Release)

  set(HALIDE_DLLS
    debug ${HALIDE_DLL_DEBUG}
    optimized ${HALIDE_DLL_RELEASE})

  mark_as_advanced(HALIDE_FOUND
    HALIDE_CMAKE_FILEPATH
    HALIDE_CMAKE_CONFIG_FILEPATH
    HALIDE_INCLUDE_DIR
    HALIDE_LIBRARIES
    HALIDE_DLLS)
else ()
  find_library(HALIDE_LIBRARIES NAMES Halide
    PATHS ${HALIDE_DISTRIB_DIR}
    PATH_SUFFIXES bin)

  # If we prefer static linking...
  # find_library(HALIDE_LIBRARIES NAMES libHalide.a Halide
  #   PATHS ${HALIDE_DISTRIB_DIR}
  #   PATH_SUFFIXES lib bin)

  mark_as_advanced(HALIDE_FOUND
    HALIDE_CMAKE_FILEPATH
    HALIDE_CMAKE_CONFIG_FILEPATH
    HALIDE_INCLUDE_DIR
    HALIDE_LIBRARIES)
endif ()

list(APPEND HALIDE_INCLUDE_DIRS
  ${HALIDE_INCLUDE_DIR}
  ${HALIDE_DISTRIB_DIR}/tools)

# Compile options
if (MSVC)
  set(HALIDE_COMPILE_OPTIONS /wd4068)
else()
  set(HALIDE_COMPILE_OPTIONS
    "-Wno-unused-but-set-variable -Wno-unused-parameter -Wno-unused-variable -Wno-missing-field-initializers -Wno-unknown-pragmas")
endif()

message("INCLUDE = ${HALIDE_INCLUDE_DIRS}")
message("LIB = ${HALIDE_LIBRARIES}")

add_library(Halide INTERFACE IMPORTED)
set_target_properties(Halide PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${HALIDE_INCLUDE_DIRS}"
  # Dynamic linking.
  INTERFACE_LINK_LIBRARIES ${HALIDE_LIBRARIES}
  # Keep this for future reference if we want to link statically.
  # INTERFACE_LINK_LIBRARIES "${HALIDE_LIBRARIES};${ZLIB_LIBRARIES}"
  INTERFACE_COMPILE_OPTIONS ${HALIDE_COMPILE_OPTIONS})

# Quick and dirty: distribute the DLLs in the binary folders.
if (WIN32)
  file(COPY ${HALIDE_DLL_DEBUG}   DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)
  file(COPY ${HALIDE_DLL_RELEASE} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)
  file(COPY ${HALIDE_DLL_RELEASE} DESTINATION ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo)
  file(COPY ${HALIDE_DLL_RELEASE} DESTINATION ${CMAKE_BINARY_DIR}/bin/MinSizeRel)
endif ()
