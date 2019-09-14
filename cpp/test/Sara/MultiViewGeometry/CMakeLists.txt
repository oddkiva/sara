project(DO_MultiViewGeometry_test)

message(STATUS "  - DO_MultiViewGeometry_Tests")

find_package(HDF5 COMPONENTS CXX REQUIRED)
find_package(DO_Sara COMPONENTS MultiViewGeometry REQUIRED)
find_package(Ceres)

file(GLOB test_multiviewgeometry_SOURCE_FILES FILES test_*.cpp)
if (NOT Ceres_FOUND)
  file(GLOB ceres_test_SOURCE_FILES FILES *ceres*.cpp)
  list(REMOVE_ITEM test_multiviewgeometry_SOURCE_FILES
    ${ceres_test_SOURCE_FILES})
endif ()

foreach (file ${test_multiviewgeometry_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(${filename} ${file} "${DO_Sara_LIBRARIES}" "MultiViewGeometry")
endforeach ()

if (Ceres_FOUND)
  target_include_directories(test_multiviewgeometry_ceres_solver_backend
    PRIVATE ${CERES_INCLUDE_DIRS})
  target_link_libraries(test_multiviewgeometry_ceres_solver_backend
    PRIVATE ${CERES_LIBRARIES})
endif ()
