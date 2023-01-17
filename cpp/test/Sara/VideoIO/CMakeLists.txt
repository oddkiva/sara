file(GLOB test_videoio_SOURCE_FILES FILES *.cpp)

foreach (file ${test_videoio_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES DO::Sara::VideoIO
    FOLDER VideoIO)
endforeach ()
