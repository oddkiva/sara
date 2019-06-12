# Use boost unit test framework.
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost
  COMPONENTS filesystem serialization system unit_test_framework
  REQUIRED)
include_directories(${Boost_INCLUDE_DIR})


message(STATUS "")
sara_step_message("Appending test projects:")

if (WIN32 AND SARA_USE_VLD)
  find_package(VisualLeakDetector)
  include_directories(${VLD_INCLUDE_DIR})
  link_directories(${VLD_LINK_DIRECTORIES})
endif ()

#add_subdirectory(Clustering)
add_subdirectory(Core)
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

if (CMAKE_COMPILER_IS_GNUCXX)
  find_program(LCOV_PATH lcov)
  find_program(GCOV_PATH gcov)
  find_program(GENHTML_PATH genhtml)

  if (LCOV_PATH AND GCOV_PATH AND GENHTML_PATH)
    add_custom_target(coverage
      COMMAND ${LCOV_PATH} --gcov-tool=${GCOV_PATH} --compat-libtool
                           --directory=${CMAKE_BINARY_DIR}
                           --base-directory=${DO_Sara_SOURCE_DIR}
                           --capture
                           --output-file=coverage.info
      COMMAND ${LCOV_PATH} --remove coverage.info 'usr/*'
                           --output-file coverage.info
      COMMAND ${LCOV_PATH} --remove coverage.info 'third-party/*'
                           --output-file coverage.info
      COMMAND ${LCOV_PATH} --remove coverage.info 'test/*'
                           --output-file coverage.info
      COMMAND ${LCOV_PATH} --remove coverage.info '${CMAKE_BINARY_DIR}/moc_*'
                           --output-file coverage.info
      COMMAND ${GENHTML_PATH} coverage.info
                              --output-directory=${CMAKE_BINARY_DIR}/coverage
    )
  endif ()
endif ()
