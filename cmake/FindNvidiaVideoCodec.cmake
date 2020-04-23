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
    if (WIN32)
      find_library(NvidiaVideoCodec_nvcuvid_LIBRARY
        NAMES nvcuvid
        ${${search}}
        PATH_SUFFIXES Lib/x64)

      find_library(NvidiaVideoCodec_encode_LIBRARY
        NAMES nvencodeapi
        ${${search}}
        PATH_SUFFIXES Lib/x64)
    else ()
      find_library(NvidiaVideoCodec_nvcuvid_LIBRARY
        NAMES nvcuvid
        ${${search}}
        PATH_SUFFIXES Lib/linux/stubs/x86_64)

      find_library(NvidiaVideoCodec_encode_LIBRARY
        NAMES nvidia-encode
        ${${search}}
        PATH_SUFFIXES Lib/linux/stubs/x86_64)
    endif ()
  endforeach ()
endif ()


mark_as_advanced(NvidiaVideoCodec_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NvidiaVideoCodec
  REQUIRED_VARS
  NvidiaVideoCodec_nvcuvid_LIBRARY
  NvidiaVideoCodec_encode_LIBRARY
  NvidiaVideoCodec_INCLUDE_DIR)

if (NvidiaVideoCodec_FOUND)
  set(NvidiaVideoCodec_INCLUDE_DIRS ${NvidiaVideoCodec_INCLUDE_DIR})

  if (NOT NvidiaVideoCodec_LIBRARIES)
    set(NvidiaVideoCodec_LIBRARIES
      ${NvidiaVideoCodec_nvcuvid_LIBRARY}
      ${NvidiaVideoCodec_encode_LIBRARY})
  endif ()

  if (NOT TARGET nvidia::VideoCodec)
    add_library(nvidia::VideoCodec INTERFACE IMPORTED)
    set_target_properties(nvidia::VideoCodec
      PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${NvidiaVideoCodec_INCLUDE_DIRS}
      INTERFACE_LINK_LIBRARIES "${NvidiaVideoCodec_LIBRARIES}")
  endif ()
endif ()
