message(STATUS "")
do_step_message("Appending third-party libraries:")

message(STATUS "  - Google Test")
add_subdirectory(gtest)

message(STATUS "  - FLANN")
add_subdirectory(flann)