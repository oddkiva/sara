find_package(DO_Sara COMPONENTS Features REQUIRED)

file(GLOB test_features_SOURCE_FILES FILES *.cpp)
foreach (file ${test_features_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES ${DO_Sara_LIBRARIES}
    FOLDER Features)
endforeach ()

target_include_directories(test_features_hdf5 PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(test_features_hdf5 PRIVATE ${HDF5_LIBRARIES})
