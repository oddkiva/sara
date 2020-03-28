file(GLOB test_SOURCE_FILES FILES *.cpp)

foreach (file ${test_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES ${DO_Sara_LIBRARIES} Halide
    FOLDER "DRAFTS/Halide")
endforeach ()

target_link_libraries(test_halide_enlarge
  PRIVATE
  shakti_halide_enlarge)
