if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_ImageIO_ADDED GLOBAL PROPERTY _DO_Sara_ImageIO_INCLUDED)

  if (NOT DO_Sara_ImageIO_ADDED)
    set(DO_ImageIO_THIRD_PARTY_LIBRARIES "")

    # Add the third-party image I/O libraries.
    list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES ${JPEG_LIBRARIES})
    list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES ${PNG_LIBRARIES})
    list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES ${TIFF_LIBRARIES})
    list(APPEND DO_ImageIO_THIRD_PARTY_LIBRARIES ${ZLIB_LIBRARIES})

    sara_glob_directory(${DO_Sara_SOURCE_DIR}/ImageIO)
    sara_create_common_variables("ImageIO")
    sara_set_internal_dependencies("ImageIO" "Core")
    sara_generate_library("ImageIO")

    target_compile_definitions(DO_Sara_ImageIO PRIVATE ${PNG_DEFINITIONS})
    target_include_directories(DO_Sara_ImageIO PRIVATE
      ${JPEG_INCLUDE_DIR}
      ${PNG_INCLUDE_DIRS}
      ${TIFF_INCLUDE_DIR}
      ${ZLIB_INCLUDE_DIRS})

    # Add the other necessary third-party libraries.
    target_include_directories(DO_Sara_ImageIO PUBLIC
      ${DO_Sara_INCLUDE_DIR}
      ${DO_Sara_ThirdParty_DIR}
      ${DO_Sara_ThirdParty_DIR}/eigen)

    target_link_libraries(DO_Sara_ImageIO
      PRIVATE ${DO_ImageIO_THIRD_PARTY_LIBRARIES}
      PUBLIC  easyexif)
  endif ()
endif ()
