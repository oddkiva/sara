find_package(OpenMP)

file(GLOB examples_SRC_FILES FILES *.cpp)

foreach (file ${examples_SRC_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  shakti_add_example(
    NAME shakti_${filename}

    SOURCES ${file}

    DEPENDENCIES
    ${DO_Sara_LIBRARIES}
    DO::Shakti::MultiArray
    DO::Shakti::ImageProcessing
    DO::Shakti::Utilities
    gpufilter)

  if (OpenMP_CXX_FOUND)
    target_link_libraries(shakti_${filename} OpenMP::OpenMP_CXX)
  endif ()
endforeach ()
