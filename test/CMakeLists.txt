include_directories(${gtest_DIR}/include)

message(STATUS "")
do_step_message("Appending test projects:")

# add_subdirectory(Core)
add_subdirectory(ImageDrawing)
# add_subdirectory(Graphics)
#set_property(TARGET Graphics PROPERTY FOLDER "DO Tests/Graphics")
# add_subdirectory(FileSystem)
#set_property(TARGET FileSystem PROPERTY FOLDER "DO Tests/FileSystem")

add_subdirectory(KDTree)
add_subdirectory(ImageProcessing)
add_subdirectory(Geometry)

# add_subdirectory(Features)
# add_subdirectory(FeatureDetectors)
# add_subdirectory(FeatureDescriptors)
# add_subdirectory(FeatureDetectorWrappers)

# add_subdirectory(Match)
# add_subdirectory(FeatureMatching)