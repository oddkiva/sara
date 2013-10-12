message(STATUS "")
do_step_message("Appending third-party libraries:")

message(STATUS "  - Google Test")
add_subdirectory(gtest)

message(STATUS "  - zlib")
add_subdirectory(zlib)

# message(STATUS "  - libjpeg")
# add_subdirectory(libjpeg)

# message(STATUS "  - libpng")
# add_subdirectory(libpng)

# message(STATUS "  - libtiff")
# add_subdirectory(libtiff)

message(STATUS "  - FLANN")
add_subdirectory(flann)