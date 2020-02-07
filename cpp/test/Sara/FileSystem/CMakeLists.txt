find_package(DO_Sara COMPONENTS FileSystem REQUIRED)

file(GLOB test_sfm_SOURCE_FILES FILES test_*.cpp)

foreach (file ${test_sfm_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES ${DO_Sara_LIBRARIES}
    FOLDER FileSystem)
endforeach ()
