if(NOT CMAKE_CUDA_COMPILER OR NOT TensorRT_FOUND)
  return()
endif()

file(GLOB test_tensorrt_SOURCE_FILES FILES test_*tensorrt*.cpp)

foreach(file ${test_tensorrt_SOURCE_FILES})
  get_filename_component(filename ${file} NAME_WE)

  add_executable(${filename} ${file})

  target_include_directories(${filename} PRIVATE ${CUDA_TOOLKIT_INCLUDE}
                                                 ${TensorRT_INCLUDE_DIRS})

  target_compile_definitions(${filename} PRIVATE -DBOOST_TEST_DYN_LINK)

  target_link_libraries(
    ${filename}
    PRIVATE ${Boost_LIBRARIES} #
            DO::Sara::ImageIO #
            DO::Sara::ImageProcessing #
            DO::Shakti::Cuda::MultiArray #
            DO::Shakti::Cuda::Utilities #
            DO::Shakti::Cuda::TensorRT)

  set_target_properties(
    ${filename} PROPERTIES COMPILE_FLAGS ${SARA_DEFINITIONS}
                           RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

  set_property(TARGET ${filename} #
               PROPERTY FOLDER "Tests/Shakti/CUDA/NeuralNetworks")

  add_test(NAME ${filename} COMMAND $<TARGET_FILE:${filename}>)
endforeach()

target_link_libraries(test_neuralnetworks_tensorrt_onnx_parser
                      PRIVATE TensorRT::OnnxParser)
