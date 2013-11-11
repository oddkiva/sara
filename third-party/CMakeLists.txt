message(STATUS "")
do_step_message("Appending third-party libraries:")

if (DO_BUILD_UNIT_TESTS)
  message(STATUS "  - Google Test")
  add_subdirectory(gtest)
endif ()

if (DO_BUILD_IMAGE_IO)
  message(STATUS "  - antigrain")
  add_subdirectory(antigrain)

  message(STATUS "  - zlib")
  add_subdirectory(zlib)

  message(STATUS "  - libjpeg")
  add_subdirectory(libjpeg)

  message(STATUS "  - libpng")
  add_subdirectory(libpng)

  message(STATUS "  - libtiff")
  add_subdirectory(libtiff)
endif ()

message(STATUS "  - FLANN")
add_subdirectory(flann)

# message(STATUS "  - DAISY")
# add_subdirectory(libdaisy)