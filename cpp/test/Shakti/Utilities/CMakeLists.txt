message(STATUS "  - DO::Shakti::Cuda::Utilities::Tests")

file(GLOB SOURCE_FILES FILES test_*.cpp test_*.cu)

foreach (file ${SOURCE_FILES})
  shakti_add_test(
    SOURCES ${file}
    DEPENDENCIES DO::Shakti::Cuda::Utilities)
endforeach ()
