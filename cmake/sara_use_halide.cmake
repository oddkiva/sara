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
  find_library(HALIDE_LIBRARY NAMES Halide
    PATHS ${HALIDE_DISTRIB_DIR}
    PATH_SUFFIXES bin)

  # If we prefer static linking...
  # find_library(HALIDE_LIBRARIES NAMES libHalide.a Halide
  #   PATHS ${HALIDE_DISTRIB_DIR}
  #   PATH_SUFFIXES lib bin)

  set(HALIDE_LIBRARIES ${HALIDE_LIBRARY})
  if (NOT APPLE)
    list(APPEND HALIDE_LIBRARIES dl)
  endif ()

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

# include(${HALIDE_CMAKE_CONFIG_FILEPATH})
# include(${HALIDE_CMAKE_FILEPATH})

if (WIN32)
  add_library(Halide SHARED IMPORTED)
else ()
  add_library(Halide INTERFACE IMPORTED)
endif ()

set_target_properties(Halide PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${HALIDE_INCLUDE_DIRS}"
  INTERFACE_COMPILE_OPTIONS ${HALIDE_COMPILE_OPTIONS})

if (WIN32)
  set_target_properties(Halide PROPERTIES
    IMPORTED_IMPLIB_DEBUG ${HALIDE_LIBRARY_DEBUG}
    IMPORTED_IMPLIB_RELEASE ${HALIDE_LIBRARY_RELEASE}
    IMPORTED_LOCATION_DEBUG ${HALIDE_DLL_DEBUG}
    IMPORTED_LOCATION_RELEASE ${HALIDE_DLL_RELEASE})
else ()
  set_target_properties(Halide PROPERTIES
    # Keep this for future reference if we want to link statically.
    # INTERFACE_LINK_LIBRARIES "${HALIDE_LIBRARIES};${ZLIB_LIBRARIES}"
    #
    # Dynamic linking.
    INTERFACE_LINK_LIBRARIES "${HALIDE_LIBRARIES}")
endif ()

# Quick and dirty: distribute the DLLs in the binary folders.
if (WIN32)
  file(COPY ${HALIDE_DLL_DEBUG}   DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)
  file(COPY ${HALIDE_DLL_RELEASE} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)
  file(COPY ${HALIDE_DLL_RELEASE} DESTINATION ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo)
  file(COPY ${HALIDE_DLL_RELEASE} DESTINATION ${CMAKE_BINARY_DIR}/bin/MinSizeRel)
endif ()


if (NOT SHAKTI_HALIDE_GPU_TARGETS)
  if (APPLE)
    set (SHAKTI_HALIDE_GPU_TARGETS metal)
  elseif (CUDA_FOUND)
    set (SHAKTI_HALIDE_GPU_TARGETS cuda)
  else ()
    set (SHAKTI_HALIDE_GPU_TARGETS opencl)
  endif ()
endif ()

include(${HALIDE_CMAKE_FILEPATH})

function (shakti_halide_library _source_filepath)
  get_filename_component(_source_filename ${_source_filepath} NAME_WE)
  halide_library(${_source_filename} SRCS ${_source_filepath})
endfunction ()

function (shakti_halide_gpu_library _source_filepath)
  get_filename_component(_source_filename ${_source_filepath} NAME_WE)
  halide_library(${_source_filename}
    SRCS ${_source_filepath}
    HALIDE_TARGET_FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})
endfunction ()
