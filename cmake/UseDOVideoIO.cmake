include_directories(
  ${DO_INCLUDE_DIR}
  ${DO_ThirdParty_DIR}/
  ${DO_ThirdParty_DIR}/eigen
  ${DO_ThirdParty_DIR}/ffmpeg/include
)

link_directories(${DO_ThirdParty_DIR}/ffmpeg/lib)


if (DO_USE_FROM_SOURCE)
  get_property(DO_VideoIO_ADDED GLOBAL PROPERTY _DO_VideoIO_INCLUDED)
  if (NOT DO_VideoIO_ADDED)
    do_glob_directory(${DO_SOURCE_DIR}/VideoIO)
    do_create_common_variables("VideoIO")
    do_generate_library("VideoIO")
    target_link_libraries(DO_VideoIO avcodec avdevice avformat avutil)
  endif ()
endif ()
