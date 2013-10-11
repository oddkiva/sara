message(STATUS "")
do_step_message("Appending third-party libraries:")

message(STATUS "  - Google Test 1.7.0")
add_subdirectory(gtest-1.7.0)

message(STATUS "  - FLANN 1.8.4")
add_subdirectory(flann-1.8.4)