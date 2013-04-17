include_directories(${gtest_DIR}/include)

message(STATUS "")
do_step_message("Appending test projects:")
add_subdirectory(Core)
add_subdirectory(Graphics)
add_subdirectory(ImageProcessing)
add_subdirectory(Features)