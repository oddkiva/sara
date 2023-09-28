find_path(
  WebP_INCLUDE_DIR
  NAMES decode.h encode.h
  PATHS /usr /usr/local
  PATH_SUFFIXES include/webp)

find_library(
  WebP_LIBRARY
  NAMES webp
  PATHS /usr /usr/local
  PATH_SUFFIXES lib)


mark_as_advanced(WebP_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  WebP
  REQUIRED_VARS WebP_LIBRARY WebP_INCLUDE_DIR)

if(WebP_FOUND)
  set(WebP_INCLUDE_DIRS ${WebP_INCLUDE_DIR})

  if(NOT WebP_LIBRARIES)
    set(WebP_LIBRARIES ${WebP_LIBRARY})
  endif()

  if(NOT TARGET WebP::WebP)
    add_library(WebP::WebP UNKNOWN IMPORTED)
    set_target_properties(WebP::WebP PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                ${WebP_INCLUDE_DIRS})
    set_property(
      TARGET WebP::WebP
      APPEND
      PROPERTY IMPORTED_LOCATION ${WebP_LIBRARY})
  endif()
endif()
