if (FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIR)
  # in cache already
  set(FFMPEG_FOUND TRUE)

elseif (MSVC)
  if (SARA_USE_FROM_SOURCE)
    set(FFMPEG_INCLUDE_DIR ${DO_Sara_ThirdParty_DIR}/ffmpeg/include)
    set(FFMPEG_LINK_DIR ${DO_Sara_ThirdParty_DIR}/ffmpeg/lib)
  else ()
    set(FFMPEG_INCLUDE_DIR ${DO_Sara_DIR}/../../../include)
    set(FFMPEG_LINK_DIR ${DO_Sara_DIR}/../../../lib)
  endif ()

  set(FFMPEG_LIBRARIES avcodec avformat avutil)

else ()

  if (SARA_FFMPEG_DIR)
    set(ENV{PKG_CONFIG_PATH} "${SARA_FFMPEG_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  endif ()

  # use pkg-config to get the directories and then use these values
  # in the FIND_PATH() and FIND_LIBRARY() calls
  find_package(PkgConfig)
  if (PKG_CONFIG_FOUND)
    pkg_check_modules(_FFMPEG_AVCODEC REQUIRED libavcodec>=55.28.1)
    pkg_check_modules(_FFMPEG_AVFORMAT libavformat)
    pkg_check_modules(_FFMPEG_AVUTIL libavutil)
  endif (PKG_CONFIG_FOUND)

  find_path(FFMPEG_AVCODEC_INCLUDE_DIR
    NAMES libavcodec/avcodec.h
    PATHS ${_FFMPEG_AVCODEC_INCLUDE_DIRS}
          /usr/include /usr/local/include /opt/local/include /sw/include
    PATH_SUFFIXES ffmpeg libav)

  find_library(FFMPEG_LIBAVCODEC
    NAMES avcodec
    PATHS ${_FFMPEG_AVCODEC_LIBRARY_DIRS}
          /usr/lib /usr/local/lib /opt/local/lib /sw/lib)

  find_library(FFMPEG_LIBAVFORMAT
    NAMES avformat
    PATHS ${_FFMPEG_AVFORMAT_LIBRARY_DIRS}
          /usr/lib /usr/local/lib /opt/local/lib /sw/lib)

  find_library(FFMPEG_LIBAVUTIL
    NAMES avutil
    PATHS ${_FFMPEG_AVUTIL_LIBRARY_DIRS}
          /usr/lib /usr/local/lib /opt/local/lib /sw/lib)

  if (FFMPEG_LIBAVCODEC AND FFMPEG_LIBAVFORMAT)
    set(FFMPEG_FOUND TRUE)
  endif()

  if (FFMPEG_FOUND)
    set(FFMPEG_INCLUDE_DIR ${FFMPEG_AVCODEC_INCLUDE_DIR})

    set(FFMPEG_LIBRARIES
      ${FFMPEG_LIBAVCODEC}
      ${FFMPEG_LIBAVFORMAT}
      ${FFMPEG_LIBAVUTIL})
  endif (FFMPEG_FOUND)

  if (FFMPEG_FOUND)
    if (NOT FFMPEG_FIND_QUIETLY)
      message(STATUS "Found FFMPEG or Libav: ${FFMPEG_LIBRARIES}, ${FFMPEG_INCLUDE_DIR}")
    endif (NOT FFMPEG_FIND_QUIETLY)
  else (FFMPEG_FOUND)
    if (FFMPEG_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find libavcodec or libavformat or libavutil")
    endif (FFMPEG_FIND_REQUIRED)
  endif (FFMPEG_FOUND)

endif ()
