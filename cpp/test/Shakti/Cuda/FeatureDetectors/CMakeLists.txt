file(GLOB test_FILES FILES test_*.cpp test_*.cu)

foreach(file ${test_FILES})
  shakti_add_test(
    SOURCES ${file}
    DEPENDENCIES DO::Sara::Core
                 DO::Shakti::Cuda::MultiArray
                 DO::Shakti::Cuda::Utilities
                 DO::Shakti::Cuda::FeatureDetectors
                 fmt::fmt
    FOLDER CUDA)
endforeach()
