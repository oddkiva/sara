find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

file(GLOB examples_SRC_FILES FILES *.cpp)

foreach (file ${examples_SRC_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  shakti_add_example(
    NAME shakti_${filename}

    SOURCES ${file}

    DEPENDENCIES
    ${DO_Sara_LIBRARIES}
    DO::Shakti::ImageProcessing
    DO::Shakti::Utilities
    gpufilter)
endforeach ()
