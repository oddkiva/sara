# You still need this cumbersome thing to make the tests pass.
find_package(DO_Sara REQUIRED)

project(DO_Sara_FileSystem_Tests)

file(GLOB test_filesystem_SOURCE_FILES FILES test_*.cpp)

foreach (file ${test_filesystem_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES DO::Sara::FileSystem
    FOLDER FileSystem)
endforeach ()
