# Find JPEG, PNG, TIFF libraries.

set(IMAGE_IO_LIBRARIES JPEG PNG TIFF ZLIB)

foreach (IMAGE_IO_LIB ${IMAGE_IO_LIBRARIES})

  if (WIN32)

    find_library(${IMAGE_IO_LIB}_DEBUG_LIBRARY
      NAMES ${IMAGE_IO_LIB}-d
      PATHS "C:/Program Files/DO-Sara-Debug/lib")

    find_library(${IMAGE_IO_LIB}_RELEASE_LIBRARY
      NAMES ${IMAGE_IO_LIB}
      PATHS "C:/Program Files/DO-Sara/lib")

    if (NOT DO_USE_STATIC_LIBS AND NOT ${IMAGE_IO_LIB}_DEBUG_LIBRARY)
      set(${IMAGE_IO_LIB}_LIBRARY ${${IMAGE_IO_LIB}_RELEASE_LIBRARY})
    else ()
      set(${IMAGE_IO_LIB}_LIBRARY
          debug ${${IMAGE_IO_LIB}_DEBUG_LIBRARY}
          optimized ${${IMAGE_IO_LIB}_RELEASE_LIBRARY})
    endif ()

  else ()
    find_package(JPEG REQUIRED)
    find_package(PNG REQUIRED)
    find_package(TIFF REQUIRED)
    find_package(ZLIB REQUIRED)
  endif ()

  set(ThirdPartyImageIO_LIBRARIES
      ${JPEG_LIBRARY} ${PNG_LIBRARY} ${TIFF_LIBRARY} ${ZLIB_LIBRARY})

endforeach()
