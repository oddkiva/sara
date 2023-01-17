project(DO_Sara_Core_Tests)

file(GLOB test_core_SOURCE_FILES FILES *.cpp)
if (NOT HDF5_FOUND)
  list(REMOVE_ITEM ${CMAKE_CURRENT_SOURCE_DIR}/test_core_hdf5.cpp)
endif ()

foreach (file ${test_core_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  if (${filename} STREQUAL "test_core_expression")
    continue ()
  endif ()

  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES DO::Sara::Core
    FOLDER Core)
endforeach ()

if (UNIX AND NOT APPLE)
  target_link_libraries(test_core_ipc_cond1 PRIVATE rt pthread)
endif ()

if (HDF5_FOUND)
  target_link_libraries(test_core_hdf5 PRIVATE ${HDF5_CXX_LIBRARIES})
endif ()
