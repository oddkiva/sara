if (NOT SARA_BUILD_VIDEOIO)
  return ()
endif ()

file(GLOB neuralnetworks_SOURCE_FILES FILES *.cpp)

foreach (file ${neuralnetworks_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  add_executable(${filename} ${filename}.cpp)
  target_link_libraries(${filename}
    PRIVATE
    DO::Sara::Graphics
    DO::Sara::ImageIO
    DO::Sara::VideoIO
    DO::Sara::Darknet
    Boost::filesystem
    $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)
  set_target_properties(${filename}
      PROPERTIES
      COMPILE_FLAGS ${SARA_DEFINITIONS}
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set_property(TARGET ${filename}
    PROPERTY
    FOLDER "Examples/Sara/NeuralNetworks")
endforeach ()
