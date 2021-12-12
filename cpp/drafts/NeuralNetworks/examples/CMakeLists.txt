find_package(DO_Sara COMPONENTS Core Graphics ImageIO REQUIRED)

file(GLOB neuralnetworks_SOURCE_FILES FILES *.cpp)

foreach (file ${neuralnetworks_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES ${DO_Sara_LIBRARIES}
    FOLDER "DRAFTS/NeuralNetworks")
endforeach ()
