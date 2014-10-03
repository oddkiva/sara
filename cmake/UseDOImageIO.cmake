include_directories(
  ${DO_INCLUDE_DIR}
  ${DO_ThirdParty_DIR}/
  ${DO_ThirdParty_DIR}/eigen
  ${DO_ThirdParty_DIR}/libjpeg
  ${DO_ThirdParty_DIR}/libpng
  ${DO_ThirdParty_DIR}/libtiff
  ${DO_ThirdParty_DIR}/zlib)

if (DO_USE_FROM_SOURCE)
  get_property(DO_ImageIO_ADDED GLOBAL PROPERTY _DO_ImageIO_INCLUDED)
  if (NOT DO_ImageIO_ADDED)
    do_glob_directory(${DO_SOURCE_DIR}/ImageIO)
    do_set_internal_dependencies("ImageIO" "easyexif;jpeg;png;tiff;zlib")
    do_create_common_variables("ImageIO")
    do_generate_library("ImageIO")
  endif ()
endif ()