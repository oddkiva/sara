include_directories(${gtest_DIR}/include)

message(STATUS "")
do_step_message("Appending test projects:")

add_subdirectory(Core)
#add_subdirectory(ImageDrawing)
add_subdirectory(Graphics)
add_subdirectory(FileSystem)

#add_subdirectory(KDTree)
add_subdirectory(ImageProcessing)
#add_subdirectory(MSER)
#add_subdirectory(Geometry)

#add_subdirectory(Features)
#add_subdirectory(FeatureDetectors)
#add_subdirectory(FeatureDescriptors)

#add_subdirectory(Match)
#add_subdirectory(FeatureMatching)

#add_subdirectory(VideoCapture)
#add_subdirectory(ChamferDistance)
