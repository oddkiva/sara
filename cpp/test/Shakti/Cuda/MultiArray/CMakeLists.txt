file(GLOB test_multiarray_SOURCE_FILES FILES test_*.cu)

foreach(file ${test_multiarray_SOURCE_FILES})
  shakti_add_test(
    SOURCES
    ${file}
    DEPENDENCIES
    DO::Sara::Core
    DO::Sara::ImageProcessing
    DO::Shakti::Cuda::MultiArray
    DO::Shakti::Cuda::Utilities)
endforeach()
