find_package(DO_Sara COMPONENTS Core ImageProcessing REQUIRED)

file(GLOB test_imageprocessing_SOURCE_FILES FILES test_*.cpp test_*.cu)

foreach (file ${test_imageprocessing_SOURCE_FILES})
  shakti_add_test(
    SOURCES
    ${file}

    DEPENDENCIES
    ${DO_Sara_LIBRARIES}
    DO::Shakti::Cuda::MultiArray
    DO::Shakti::Cuda::Utilities
    DO::Shakti::Cuda::ImageProcessing)
endforeach ()
