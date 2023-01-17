file(GLOB test_multiviewgeometry_SOURCE_FILES FILES test_*.cpp)
file(GLOB test_ceres_SOURCE_FILES FILES test_*ceres*.cpp)

# Remove tests depending on Ceres if it is not found.
if (NOT Ceres_FOUND)
  sara_message("Cannot testing Ceres Solver backend.")
  foreach (file ${test_ceres_SOURCE_FILES})
    list(REMOVE_ITEM test_multiviewgeometry_SOURCE_FILES ${file})
  endforeach()
else ()
  sara_message("Enabling testing with Ceres Solver backend.")
endif ()

foreach (file ${test_multiviewgeometry_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES DO::Sara::MultiViewGeometry fmt::fmt
    FOLDER MultiViewGeometry)
endforeach ()


# Link tests with Ceres if it is found.
if (Ceres_FOUND)
  sara_message("Adding tests using Ceres Solver backend.")
  foreach (file ${test_ceres_SOURCE_FILES})
    get_filename_component(filename "${file}" NAME_WE)
    target_include_directories(${filename} PRIVATE ${CERES_INCLUDE_DIRS})
    target_link_libraries(${filename} PRIVATE ${CERES_LIBRARIES})
  endforeach()
endif ()
