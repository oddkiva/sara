message(STATUS "")
sara_step_message("Appending test projects:")

add_subdirectory(Core)
add_subdirectory(FileSystem)
add_subdirectory(Graphics)
add_subdirectory(ImageIO)
if (SARA_BUILD_VIDEOIO)
  add_subdirectory(VideoIO)
endif ()

add_subdirectory(Clustering)
add_subdirectory(DisjointSets)

add_subdirectory(KDTree)
add_subdirectory(ImageProcessing)
add_subdirectory(Geometry)

add_subdirectory(Features)
add_subdirectory(FeatureDetectors)
add_subdirectory(FeatureDescriptors)

add_subdirectory(Match)
add_subdirectory(FeatureMatching)
add_subdirectory(MultiViewGeometry)
