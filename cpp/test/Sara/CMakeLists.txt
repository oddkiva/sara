# Use boost unit test framework.
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost
  COMPONENTS filesystem serialization system unit_test_framework
  REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
add_definitions(-DBOOST_ALL_NO_LIB)


message(STATUS "")
sara_step_message("Appending test projects:")

if (WIN32 AND SARA_USE_VLD)
  find_package(VisualLeakDetector)
  include_directories(${VLD_INCLUDE_DIR})
  link_directories(${VLD_LINK_DIRECTORIES})
endif ()

add_subdirectory(Core)
add_subdirectory(FileSystem)
add_subdirectory(Graphics)
add_subdirectory(ImageIO)
if (SARA_BUILD_VIDEOIO)
  add_subdirectory(VideoIO)
endif ()

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
