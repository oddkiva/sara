file(GLOB TRT_SOURCE_FILES FILES *.cpp)

foreach(file ${TRT_SOURCE_FILES})
  get_filename_component(filename ${file} NAME_WE)

  add_executable(${filename} ${file})

  target_include_directories(${filename} PRIVATE ${CUDA_TOOLKIT_INCLUDE}
                                                 ${TensorRT_INCLUDE_DIRS})

  target_link_libraries(
    ${filename}
    PRIVATE ${Boost_LIBRARIES} #
            DO::Sara::Graphics
            DO::Sara::ImageIO
            DO::Sara::VideoIO
            DO::Sara::ImageProcessing
            DO::Sara::Darknet
            DO::Sara::TensorRT
            DO::Shakti::Cuda::MultiArray
            DO::Shakti::Cuda::Utilities)

  set_target_properties(
    ${filename} PROPERTIES COMPILE_FLAGS ${SARA_DEFINITIONS}
                           RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

  set_property(TARGET ${filename} PROPERTY FOLDER
                                           "Examples/Shakti/NeuralNetworks")
endforeach()
