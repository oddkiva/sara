find_package(DO_Sara COMPONENTS Visualization REQUIRED)

file(GLOB test_visualization_SOURCE_FILES FILES *.cpp)
foreach (file ${test_visualization_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES ${DO_Sara_LIBRARIES}
    FOLDER Features)
endforeach ()
