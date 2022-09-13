file(GLOB test_neuralnetworks_SOURCE_FILES FILES test_*.cpp)

foreach(file ${test_neuralnetworks_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES DO::Sara::Darknet
    FOLDER "NeuralNetworks")
endforeach()
