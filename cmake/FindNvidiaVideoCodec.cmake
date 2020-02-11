# This module defines the following variables:
#
# ::
#
#   NvidiaVideoCodec_INCLUDE_DIRS
#   NvidiaVideoCodec_LIBRARIES
#   NvidiaVideoCodec_FOUND
#
# ::
#
#   NvidiaVideoCodec_VERSION_STRING - version (x.y.z)
#   NvidiaVideoCodec_VERSION_MAJOR  - major version (x)
#   NvidiaVideoCodec_VERSION_MINOR  - minor version (y)
#   NvidiaVideoCodec_VERSION_PATCH  - patch version (z)
#
# Hints
# ^^^^^
# A user may set ``NvidiaVideoCodec_ROOT`` to an installation root to tell this
# module where to look.
#
set(_NvidiaVideoCodec_SEARCHES)

if (NvidiaVideoCodec_ROOT)
  set(_NvidiaVideoCodec_SEARCH_ROOT PATHS ${NvidiaVideoCodec_ROOT} NO_DEFAULT_PATH)
  list(APPEND _NvidiaVideoCodec_SEARCHES _NvidiaVideoCodec_SEARCH_ROOT)
endif ()

# Appends some common paths
set (_NvidiaVideoCodec_SEARCH_NORMAL PATHS "/usr" "/opt")
list(APPEND _NvidiaVideoCodec_SEARCHES _NvidiaVideoCodec_SEARCH_NORMAL)

# Include dir
foreach (search ${_NvidiaVideoCodec_SEARCHES})
  find_path(NvidiaVideoCodec_INCLUDE_DIR
    NAMES nvcuvid.h ${${search}}
    PATH_SUFFIXES include)
endforeach ()

if (NOT NvidiaVideoCodec_LIBRARY)
  foreach (search ${_NvidiaVideoCodec_SEARCHES})
    find_library(NvidiaVideoCodec_LIBRARY
      NAMES nvcuvid ${${search}}
      PATH_SUFFIXES Lib/linux/stubs/x86_64)
  endforeach ()
endif ()


mark_as_advanced(NvidiaVideoCodec_INCLUDE_DIR)

# if (NvidiaVideoCodec_INCLUDE_DIR AND
#     EXISTS "${NvidiaVideoCodec_INCLUDE_DIR}/nvcuvid.h")
#   file(STRINGS "${NvidiaVideoCodec_INCLUDE_DIR}/nvcuvid.h"
#     NvidiaVideoCodec_MAJOR REGEX "^#define NV_VIDEOCODEC_MAJOR [0-9]+.*$")
#   file(STRINGS "${NvidiaVideoCodec_INCLUDE_DIR}/nvcuvid.h"
#     NvidiaVideoCodec_MINOR REGEX "^#define NV_VIDEOCODEC_MINOR [0-9]+.*$")
#   file(STRINGS "${NvidiaVideoCodec_INCLUDE_DIR}/nvcuvid.h" NvidiaVideoCodec_PATCH REGEX
#     "^#define NV_VIDEOCODEC_PATCH [0-9]+.*$")
#
#   string(REGEX REPLACE "^#define NV_VIDEOCODEC_MAJOR ([0-9]+).*$" "\\1" NvidiaVideoCodec_VERSION_MAJOR "${NvidiaVideoCodec_MAJOR}")
#   string(REGEX REPLACE "^#define NV_VIDEOCODEC_MINOR ([0-9]+).*$" "\\1" NvidiaVideoCodec_VERSION_MINOR "${NvidiaVideoCodec_MINOR}")
#   string(REGEX REPLACE "^#define NV_VIDEOCODEC_PATCH ([0-9]+).*$" "\\1" NvidiaVideoCodec_VERSION_PATCH "${NvidiaVideoCodec_PATCH}")
#   set(NvidiaVideoCodec_VERSION_STRING "${NvidiaVideoCodec_VERSION_MAJOR}.${NvidiaVideoCodec_VERSION_MINOR}.${NvidiaVideoCodec_VERSION_PATCH}")
# endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NvidiaVideoCodec
  REQUIRED_VARS
  NvidiaVideoCodec_LIBRARY
  NvidiaVideoCodec_INCLUDE_DIR)
  # VERSION_VAR
  # NvidiaVideoCodec_VERSION_STRING)

if (NvidiaVideoCodec_FOUND)
  set(NvidiaVideoCodec_INCLUDE_DIRS ${NvidiaVideoCodec_INCLUDE_DIR})

  if (NOT NvidiaVideoCodec_LIBRARIES)
    set(NvidiaVideoCodec_LIBRARIES ${NvidiaVideoCodec_LIBRARY})
  endif ()

  if (NOT TARGET nvidia::VideoCodec)
    add_library(nvidia::VideoCodec INTERFACE IMPORTED)
    set_target_properties(nvidia::VideoCodec
      PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${NvidiaVideoCodec_INCLUDE_DIRS}
      INTERFACE_LINK_LIBRARIES ${NvidiaVideoCodec_LIBRARY})
  endif ()
endif ()
