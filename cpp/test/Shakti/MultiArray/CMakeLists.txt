find_package(DO_Sara COMPONENTS Core ImageProcessing REQUIRED)

file(GLOB test_multiarray_SOURCE_FILES FILES test_*.cu)

foreach (file ${test_multiarray_SOURCE_FILES})
  shakti_add_test(
    SOURCES
    ${file}

    DEPENDENCIES
    DO::Shakti::Cuda::MultiArray
    DO::Shakti::Cuda::Utilities)
endforeach ()
