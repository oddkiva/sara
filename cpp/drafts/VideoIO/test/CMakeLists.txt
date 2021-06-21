file(GLOB test_AudioIO_SOURCE_FILES FILES test_*.cpp)

foreach (file ${test_AudioIO_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}

    DEPENDENCIES
    ${DO_Sara_LIBRARIES}
    DO_Sara_AudioIO

    FOLDER AudioIO)
endforeach ()
