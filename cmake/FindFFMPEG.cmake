if (FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIR)
  set(FFMPEG_FOUND TRUE)

elseif (WIN32)
  # TODO: make it better but for now just trust vcpkg.
  set(FFMPEG_LIBRARIES avcodec avformat avutil swscale)
else ()
  # use pkg-config to get the directories and then use these values
  # in the FIND_PATH() and FIND_LIBRARY() calls
  find_package(PkgConfig)
  if (PKG_CONFIG_FOUND)
    pkg_check_modules(_FFMPEG_AVCODEC REQUIRED libavcodec>=55.28.1)
    pkg_check_modules(_FFMPEG_AVFORMAT libavformat)
    pkg_check_modules(_FFMPEG_AVUTIL libavutil)
    pkg_check_modules(_FFMPEG_SWSCALE libswscale)
  endif (PKG_CONFIG_FOUND)

  find_path(FFMPEG_AVCODEC_INCLUDE_DIR
    NAMES libavcodec/avcodec.h
    PATHS ${_FFMPEG_AVCODEC_INCLUDE_DIRS}
          /usr/include /usr/local/include /opt/local/include
    PATH_SUFFIXES ffmpeg libav)

  find_library(FFMPEG_LIBAVCODEC
    NAMES avcodec
    PATHS ${_FFMPEG_AVCODEC_LIBRARY_DIRS}
          /usr/lib /usr/local/lib /opt/local/lib)

  find_library(FFMPEG_LIBAVFORMAT
    NAMES avformat
    PATHS ${_FFMPEG_AVFORMAT_LIBRARY_DIRS}
          /usr/lib /usr/local/lib /opt/local/lib)

  find_library(FFMPEG_LIBAVUTIL
    NAMES avutil
    PATHS ${_FFMPEG_AVUTIL_LIBRARY_DIRS}
          /usr/lib /usr/local/lib /opt/local/lib)

  find_library(FFMPEG_LIBSWSCALE
    NAMES swscale
    PATHS ${_FFMPEG_AVUTIL_LIBRARY_DIRS}
          /usr/lib /usr/local/lib /opt/local/lib)

  if (FFMPEG_LIBAVCODEC AND FFMPEG_LIBAVFORMAT AND FFMPEG_LIBAVUTIL AND
      FFMPEG_LIBSWSCALE)
    set(FFMPEG_FOUND TRUE)
  endif()

  if (FFMPEG_FOUND)
    set(FFMPEG_INCLUDE_DIR ${FFMPEG_AVCODEC_INCLUDE_DIR})

    set(FFMPEG_LIBRARIES
      ${FFMPEG_LIBAVCODEC}
      ${FFMPEG_LIBAVFORMAT}
      ${FFMPEG_LIBAVUTIL}
      ${FFMPEG_LIBSWSCALE})
  else ()
      message(FATAL_ERROR "Could not find all FFmpeg libraries!")
  endif ()
endif ()
