message(STATUS "")
do_step_message("Appending third-party libraries:")

message(STATUS "  - Google Test 1.7.0")
add_subdirectory(gtest-1.7.0)

message(STATUS "  - zlib 1.2.8")
add_subdirectory(zlib-1.2.8)

message(STATUS "  - libjpeg 6b")
add_subdirectory(libjpeg-6b)

message(STATUS "  - libpng 1.6.5")
add_subdirectory(libpng-1.6.5)

message(STATUS "  - libtiff 4.0.3")
add_subdirectory(libtiff-4.0.3)

message(STATUS "  - FLANN 1.8.4")
add_subdirectory(flann-1.8.4)