find_package(JPEG REQUIRED)
find_package(PNG  REQUIRED)
find_package(TIFF REQUIRED)
find_package(ZLIB REQUIRED)

set(DO_ImageIO_THIRD_PARTY_LIBRARIES "")

# Add the third-party image I/O libraries.
include_directories(${JPEG_INCLUDE_DIR})
list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES ${JPEG_LIBRARIES})

add_definitions(${PNG_DEFINITIONS})
include_directories(${PNG_INCLUDE_DIRS})
list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES ${PNG_LIBRARIES})

include_directories(${TIFF_INCLUDE_DIR})
list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES ${TIFF_LIBRARIES})

include_directories(${ZLIB_INCLUDE_DIRS})
list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES ${ZLIB_LIBRARIES})

# Add the other necessary third-party libraries.
include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_Sara_ThirdParty_DIR}
  ${DO_Sara_ThirdParty_DIR}/eigen)


if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_ImageIO_ADDED GLOBAL PROPERTY _DO_Sara_ImageIO_INCLUDED)
  if (NOT DO_Sara_ImageIO_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/ImageIO)
    sara_create_common_variables("ImageIO")
    sara_set_internal_dependencies(
      "ImageIO" "Core")
    sara_generate_library("ImageIO")

    target_link_libraries(DO_Sara_ImageIO
      PRIVATE ${DO_ImageIO_THIRD_PARTY_LIBRARIES}
      PUBLIC  easyexif)
  endif ()
endif ()
