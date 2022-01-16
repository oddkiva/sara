find_package(OpenMP)

file(GLOB examples_SRC_FILES FILES *.cpp)

foreach (file ${examples_SRC_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  shakti_add_example(
    NAME shakti_${filename}

    SOURCES ${file}

    DEPENDENCIES
    CUDA::cudart
    ${DO_Sara_LIBRARIES}
    DO::Shakti::Cuda::MultiArray
    DO::Shakti::Cuda::ImageProcessing
    DO::Shakti::Cuda::Utilities
    $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)
  set_target_properties(
    shakti_${filename}
    PROPERTIES
    FOLDER "Examples/Shakti/Cuda")
endforeach ()
