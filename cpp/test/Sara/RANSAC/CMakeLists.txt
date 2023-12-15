file(GLOB test_SOURCE_FILES FILES test_*.cpp)

foreach (file ${test_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES DO::Sara::MultiViewGeometry DO::Sara::RANSAC
    FOLDER RANSAC)
endforeach ()
