file(GLOB test_imageprocessing_SOURCE_FILES FILES test_*.cpp)

foreach (file ${test_imageprocessing_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES DO::Sara::ImageProcessing
    FOLDER ImageProcessing)
endforeach ()
