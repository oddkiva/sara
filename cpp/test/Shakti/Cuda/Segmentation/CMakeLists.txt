find_package(DO_Sara COMPONENTS Core ImageProcessing REQUIRED)


file(GLOB test_imageprocessing_SOURCE_FILES FILES test_*.cu)

foreach (file ${test_imageprocessing_SOURCE_FILES})
  shakti_add_test(
    SOURCES ${file}
    DEPENDENCIES DO::Shakti::Cuda::Utilities DO::Shakti::Cuda::Segmentation)
endforeach ()
