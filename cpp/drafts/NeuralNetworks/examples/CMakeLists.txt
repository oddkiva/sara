find_package(DO_Sara COMPONENTS Core Graphics ImageProcessing ImageIO VideoIO REQUIRED)

file(GLOB neuralnetworks_SOURCE_FILES FILES *.cpp)

foreach (file ${neuralnetworks_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  add_executable(${filename} ${filename}.cpp)
  target_link_libraries(${filename}
    PRIVATE
    ${DO_Sara_LIBRARIES}
    ${Boost_LIBRARIES}
    $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)
  set_target_properties(${filename}
      PROPERTIES
      COMPILE_FLAGS ${SARA_DEFINITIONS}
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set_property(TARGET ${filename}
    PROPERTY
    FOLDER "Examples/Sara/NeuralNetworks")
endforeach ()
