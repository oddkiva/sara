project(DO_SfM_test)

message(STATUS "  - DO_SfM_Tests")

find_package(DO_Sara COMPONENTS FileSystem REQUIRED)

file(GLOB test_sfm_SOURCE_FILES FILES test_*.cpp)

foreach (file ${test_sfm_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(${filename} ${file} "${DO_Sara_LIBRARIES}" "FileSystem")
  target_compile_definitions(${filename} PRIVATE -DBOOST_FILESYSTEM_DYN_LINK)
endforeach ()
