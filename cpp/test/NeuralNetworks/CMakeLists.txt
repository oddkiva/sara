project(DO_MultiViewGeometry_test)

message(STATUS "  - DO_NeuralNetworks_Tests")

find_package(DO_Sara COMPONENTS Core REQUIRED)

file(GLOB test_neuralnetworks_SOURCE_FILES FILES test_*.cpp)

foreach (file ${test_neuralnetworks_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(${filename} ${file} "${DO_Sara_LIBRARIES}")
endforeach ()
