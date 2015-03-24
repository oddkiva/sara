include_directories(
  ${DO_INCLUDE_DIR}
  ${DO_ThirdParty_DIR}/
  ${DO_ThirdParty_DIR}/eigen
)

find_package(FFMPEG REQUIRED)

include_directories(${FFMPEG_INCLUDE_DIR})
if (MSVC)
  link_directories(${FFMPEG_LINK_DIR})
endif ()


if (DO_USE_FROM_SOURCE)
  get_property(DO_VideoIO_ADDED GLOBAL PROPERTY _DO_VideoIO_INCLUDED)
  if (NOT DO_VideoIO_ADDED)
    do_glob_directory(${DO_SOURCE_DIR}/VideoIO)
    do_create_common_variables("VideoIO")
    do_generate_library("VideoIO")
    target_link_libraries(DO_VideoIO ${FFMPEG_LIBRARIES})
  endif ()
endif ()
