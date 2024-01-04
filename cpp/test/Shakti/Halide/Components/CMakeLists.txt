file(GLOB test_SOURCE_FILES FILES *.cpp)

foreach (file ${test_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES DO::Sara::Core
                 Halide::Halide
                 ${CMAKE_DL_LIBS})
  set_target_properties(${filename}
    PROPERTIES
    FOLDER "Tests/Shakti/Halide")
endforeach ()
