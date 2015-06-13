find_package(JPEG QUIET)
find_package(PNG QUIET)
find_package(TIFF QUIET)
find_package(ZLIB QUIET)

set(DO_ImageIO_THIRD_PARTY_LIBRARIES "")

# Add the third-party image I/O libraries.
if (JPEG_FOUND)
  include_directories(${JPEG_INCLUDE_DIR})
  list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES ${JPEG_LIBRARIES})
else ()
  include_directories(${DO_Sara_ThirdParty_DIR}/libjpeg)
  list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES jpeg)
endif ()

if (PNG_FOUND)
  add_definitions(${PNG_DEFINITIONS})
  include_directories(${PNG_INCLUDE_DIRS})
  list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES ${PNG_LIBRARIES})
else ()
  include_directories(${DO_Sara_ThirdParty_DIR}/libpng)
  list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES png)
endif ()

if (TIFF_FOUND)
  include_directories(${TIFF_INCLUDE_DIR})
  list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES ${TIFF_LIBRARIES})
else ()
  include_directories(${DO_Sara_ThirdParty_DIR}/libtiff)
  list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES tiff)
endif ()

if (ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES ${ZLIB_LIBRARIES})
else ()
  include_directories(${DO_Sara_ThirdParty_DIR}/zlib)
  list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES zlib)
endif ()

# Add the other necessary third-party libraries.
include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_Sara_ThirdParty_DIR}
  ${DO_Sara_ThirdParty_DIR}/eigen
)


if (DO_USE_FROM_SOURCE)
  get_property(DO_Sara_ImageIO_ADDED GLOBAL PROPERTY _DO_Sara_ImageIO_INCLUDED)
  if (NOT DO_Sara_ImageIO_ADDED)
    do_glob_directory(${DO_Sara_SOURCE_DIR}/ImageIO)
    do_create_common_variables("ImageIO")
    do_generate_library("ImageIO")
    target_link_libraries(DO_Sara_ImageIO easyexif ${DO_ImageIO_THIRD_PARTY_LIBRARIES})
  endif ()
endif ()
