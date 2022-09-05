find_path(
  HEIF_INCLUDE_DIR
  NAMES heif.h heif_cxx.h heif_version.h
  PATHS /usr /usr/local
  PATH_SUFFIXES include/libheif)

find_library(
  HEIF_LIBRARY
  NAMES heif
  PATHS /usr /usr/local
  PATH_SUFFIXES lib)

if(HEIF_INCLUDE_DIR AND EXISTS "${HEIF_INCLUDE_DIR}/heif_version.h")
  file(STRINGS "${HEIF_INCLUDE_DIR}/heif_version.h" HEIF_VERSION_STRING
       REGEX "^#define LIBHEIF_VERSION [0-9]+.*$")
endif()

mark_as_advanced(HEIF_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  HEIF
  REQUIRED_VARS HEIF_LIBRARY HEIF_INCLUDE_DIR
  VERSION_VAR HEIF_VERSION_STRING)

if(HEIF_FOUND)
  set(HEIF_INCLUDE_DIRS ${HEIF_INCLUDE_DIR})

  if(NOT HEIF_LIBRARIES)
    set(HEIF_LIBRARIES ${HEIF_LIBRARY})
  endif()

  if(NOT TARGET HEIF::HEIF)
    add_library(HEIF::HEIF UNKNOWN IMPORTED)
    set_target_properties(HEIF::HEIF PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                ${HEIF_INCLUDE_DIRS})
    set_property(
      TARGET HEIF::HEIF
      APPEND
      PROPERTY IMPORTED_LOCATION ${HEIF_LIBRARY})
  endif()
endif()
