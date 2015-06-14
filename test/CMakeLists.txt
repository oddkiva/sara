include_directories(${DO_Sara_ThirdParty_DIR}/gtest/include)

message(STATUS "")
do_step_message("Appending test projects:")

if (WIN32 AND DO_USE_VLD)
  find_package(VisualLeakDetector)
  include_directories(${VLD_INCLUDE_DIR})
  link_directories(${VLD_LINK_DIRECTORIES})
endif ()

add_subdirectory(Core)
add_subdirectory(Graphics)
#add_subdirectory(ImageDrawing)
add_subdirectory(ImageIO)
add_subdirectory(VideoIO)

add_subdirectory(KDTree)
add_subdirectory(ImageProcessing)
add_subdirectory(Geometry)

add_subdirectory(Features)
#add_subdirectory(FeatureDetectors)
#add_subdirectory(FeatureDescriptors)

#add_subdirectory(Match)
#add_subdirectory(FeatureMatching)

#add_subdirectory(VideoCapture)
