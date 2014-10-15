include_directories(${DO_ThirdParty_DIR}/gtest/include)

message(STATUS "")
do_step_message("Appending test projects:")

add_subdirectory(Core)
add_subdirectory(Graphics)
#add_subdirectory(ImageDrawing)
add_subdirectory(ImageIO)

add_subdirectory(KDTree)
add_subdirectory(ImageProcessing)
add_subdirectory(Geometry)

#add_subdirectory(Features)
#add_subdirectory(FeatureDetectors)
#add_subdirectory(FeatureDescriptors)

#add_subdirectory(Match)
#add_subdirectory(FeatureMatching)

#add_subdirectory(VideoCapture)
