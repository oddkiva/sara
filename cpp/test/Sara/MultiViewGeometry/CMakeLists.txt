project(DO_MultiViewGeometry_test)

message(STATUS "  - DO_MultiViewGeometry_Tests")

find_package(HDF5 COMPONENTS CXX REQUIRED)
find_package(DO_Sara COMPONENTS MultiViewGeometry REQUIRED)
if (NOT WIN32)
  find_package(Ceres REQUIRED)
endif ()

file(GLOB test_multiviewgeometry_SOURCE_FILES FILES test_*.cpp)

foreach (file ${test_multiviewgeometry_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(${filename} ${file} "${DO_Sara_LIBRARIES}" "MultiViewGeometry")
endforeach ()

if (WIN32)
  target_compile_definitions(test_multiviewgeometry_ceres_solver_backend
    PRIVATE -DCERES_USING_SHARED_LIBRARY
            -DCERES_USE_OPENMP
            -DCERES_MSVC_USE_UNDERSCORE_PREFIXED_BESSEL_FUNCTIONS
            -DMAX_LOG_LEVEL=2)
  target_include_directories(test_multiviewgeometry_ceres_solver_backend
    PRIVATE ${DO_Sara_ThirdParty_DIR}/ceres-solver-1.14.0/include
            ${DO_Sara_ThirdParty_DIR}/ceres-solver-1.14.0/internal/ceres/miniglog
            ${DO_Sara_ThirdParty_DIR}/ceres-solver-1.14.0/config)
  target_link_libraries(test_multiviewgeometry_ceres_solver_backend
    PRIVATE ceres)
else ()
  target_include_directories(test_multiviewgeometry_ceres_solver_backend
    PRIVATE ${CERES_INCLUDE_DIRS})
  target_link_libraries(test_multiviewgeometry_ceres_solver_backend
    PRIVATE ${CERES_LIBRARIES})
endif ()
