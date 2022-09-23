find_package(TensorRT)

file(GLOB test_tensorrt_SOURCE_FILES FILES test_*tensorrt*.cpp)

if(CMAKE_CUDA_COMPILER AND TensorRT_FOUND)
  foreach(file ${test_tensorrt_SOURCE_FILES})
    get_filename_component(filename ${file} NAME_WE)

    add_executable(${filename} ${file})

    target_include_directories(${filename} PRIVATE ${CUDA_TOOLKIT_INCLUDE}
                                                   ${TensorRT_INCLUDE_DIRS})

    target_compile_definitions(${filename} PRIVATE -DBOOST_TEST_DYN_LINK)

    target_compile_features(${filename} PRIVATE cxx_std_17)

    target_link_libraries(
      ${filename}
      PRIVATE ${Boost_LIBRARIES} #
              $<$<PLATFORM_ID:Linux>:CUDA::cudart> #
              ${TensorRT_LIBRARIES}
              DO::Sara::ImageIO
              DO::Sara::ImageProcessing
              DO::Sara::Darknet
              DO::Shakti::Cuda::MultiArray
              DO::Shakti::Cuda::Utilities)

    set_target_properties(
      ${filename} PROPERTIES COMPILE_FLAGS ${SARA_DEFINITIONS}
                             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    set_property(TARGET ${filename} PROPERTY FOLDER
                                             "Tests/Shakti/NeuralNetworks")

    add_test(NAME ${filename} COMMAND $<TARGET_FILE:${filename}>)
  endforeach()
endif()
