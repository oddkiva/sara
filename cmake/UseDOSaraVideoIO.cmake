include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_ThirdParty_DIR}/
  ${DO_ThirdParty_DIR}/eigen
)

find_package(FFMPEG REQUIRED)

include_directories(${FFMPEG_INCLUDE_DIR})
if (MSVC)
  link_directories(${FFMPEG_LINK_DIR})
endif ()


if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_VideoIO_ADDED GLOBAL PROPERTY _DO_Sara_VideoIO_INCLUDED)
  if (NOT DO_Sara_VideoIO_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/VideoIO)
    sara_create_common_variables("VideoIO")
    sara_generate_library("VideoIO")
    target_link_libraries(DO_Sara_VideoIO ${FFMPEG_LIBRARIES})
  endif ()
endif ()
