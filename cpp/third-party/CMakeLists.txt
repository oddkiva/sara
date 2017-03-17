message(STATUS "")
sara_step_message("Appending third-party libraries:")

set(THIRDPARTY_FOLDER_NAME "DO Sara Third-Party")

# Disable any specific compiler flags for all third-party software.
set (CMAKE_C_FLAGS "")
set (CMAKE_CXX_FLAGS "")
if (UNIX AND SARA_BUILD_SHARED_LIBS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif ()


if (SARA_BUILD_TESTS)
  if (WIN32)
    file (COPY ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/bin/
          DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)
    file (COPY ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/bin/
          DESTINATION ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo)
    file (COPY ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg/bin/
          DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)
  endif ()
endif ()


# EXIF reader.
message(STATUS "  - easyexif")
add_subdirectory(easyexif)
set_property(TARGET easyexif PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})


# KD-tree.
message(STATUS "  - FLANN")
add_subdirectory(flann)
set_property(TARGET flann_cpp_s PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})


# Image I/O.
find_package(ZLIB QUIET)
find_package(JPEG QUIET)
find_package(PNG QUIET)
find_package(TIFF QUIET)

if (NOT ZLIB_FOUND)
  message(STATUS "  - zlib")
  add_subdirectory(zlib)
  set_property(TARGET zlib PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})
endif ()

if (NOT JPEG_FOUND)
  message(STATUS "  - libjpeg")
  add_subdirectory(libjpeg)
  set_property(TARGET jpeg PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})
endif ()

if (NOT PNG_FOUND)
  message(STATUS "  - libpng")
  add_subdirectory(libpng)
  set_property(TARGET png PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})
endif ()

if (NOT TIFF_FOUND)
  message(STATUS "  - libtiff")
  add_subdirectory(libtiff)
  set_property(TARGET tiff PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})
endif ()

if (WIN32)
  message(STATUS "  - FFMpeg")
  add_subdirectory(ffmpeg)
endif ()
