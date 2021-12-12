find_package(DO_Sara COMPONENTS Core Graphics ImageIO REQUIRED)

file(GLOB neuralnetworks_SOURCE_FILES FILES *.cpp)

foreach (file ${neuralnetworks_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  add_executable(${filename} ${filename}.cpp)
  target_link_libraries(${filename}
    PRIVATE
    ${DO_Sara_LIBRARIES}
    ${Boost_LIBRARIES})
  set_property(TARGET ${filename}
    PROPERTY
    FOLDER "DRAFTS/NeuralNetworks")
endforeach ()
