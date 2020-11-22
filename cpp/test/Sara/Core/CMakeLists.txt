find_package(DO_Sara COMPONENTS Core REQUIRED)

file(GLOB test_core_SOURCE_FILES FILES *.cpp)

foreach (file ${test_core_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  if (${filename} STREQUAL "test_core_expression")
    continue ()
  endif ()

  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES ${DO_Sara_LIBRARIES}
    FOLDER Core)
endforeach ()

if (UNIX AND NOT APPLE)
  target_link_libraries(test_core_ipc_cond1 PRIVATE rt pthread)
endif ()

find_package(HDF5 COMPONENTS CXX REQUIRED)
target_include_directories(test_core_hdf5 PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(test_core_hdf5 PRIVATE ${HDF5_LIBRARIES})
