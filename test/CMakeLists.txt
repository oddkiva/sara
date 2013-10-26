include_directories(${gtest_DIR}/include)

message(STATUS "")
do_step_message("Appending test projects:")
add_subdirectory(Core)
add_subdirectory(Graphics)
add_subdirectory(ImageProcessing)

add_subdirectory(Geometry)

add_subdirectory(Features)
add_subdirectory(FeatureDetectors)

#add_subdirectory(Match)
#add_subdirectory(FeatureMatching)