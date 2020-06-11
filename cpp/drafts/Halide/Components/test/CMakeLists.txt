file(GLOB test_SOURCE_FILES FILES *.cpp)

foreach (file ${test_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES Halide
    FOLDER "DRAFTS/Halide")
  set_target_properties(${filename}
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES)
endforeach ()
