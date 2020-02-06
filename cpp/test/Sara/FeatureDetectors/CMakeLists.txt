find_package(DO_Sara COMPONENTS Graphics Features FeatureDetectors REQUIRED)

set(TEST_GROUP_NAME "FeatureDetectors")
file(GLOB test_feature_detectors_SOURCE_FILES FILES test_*.cpp)
foreach (file ${test_feature_detectors_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES ${DO_Sara_LIBRARIES}
    FOLDER FeatureDetectors)
endforeach ()

