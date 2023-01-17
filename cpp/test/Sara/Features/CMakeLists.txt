file(GLOB test_features_SOURCE_FILES FILES *.cpp)
foreach (file ${test_features_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES DO::Sara::Features
    FOLDER Features)
endforeach ()

target_include_directories(test_features_hdf5 PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(test_features_hdf5 PRIVATE ${HDF5_LIBRARIES})
