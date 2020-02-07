message(STATUS "  - DO_ImageProcessing_tests")

find_package(DO_Sara COMPONENTS Core ImageProcessing REQUIRED)

file(GLOB test_imageprocessing_SOURCE_FILES FILES test_*.cpp)

set (DEPS
  ${DO_Sara_LIBRARIES}
  DO::Shakti::Utilities
  DO::Shakti::ImageProcessing)

foreach (file ${test_imageprocessing_SOURCE_FILES})
  shakti_add_test(SOURCES ${file} DEPENDENCIES ${DEPS})
endforeach ()
