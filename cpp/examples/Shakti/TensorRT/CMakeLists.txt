if(NOT CMAKE_CUDA_COMPILER OR NOT TensorRT_FOUND)
  return()
endif()

# This is a workaround with MSVC to avoid fighting with CMake and CUDA.
add_library(tensorrt_yolov4_utilities downsample.hpp downsample.cu)
target_link_libraries(tensorrt_yolov4_utilities
                      PRIVATE DO::Shakti::Cuda::TensorRT)
set_property(TARGET tensorrt_yolov4_utilities
             PROPERTY FOLDER "Examples/Shakti/NeuralNetworks")

list(APPEND TRT_EXAMPLE_FILES tensorrt_yolov4_example.cpp)

foreach(file ${TRT_EXAMPLE_FILES})
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
            DO::Shakti::Cuda::MultiArray
            DO::Shakti::Cuda::Utilities
            DO::Shakti::Cuda::TensorRT
            tinycolormap
            tensorrt_yolov4_utilities)

  set_target_properties(
    ${filename} PROPERTIES COMPILE_FLAGS ${SARA_DEFINITIONS}
                           RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

  set_property(TARGET ${filename} PROPERTY FOLDER
                                           "Examples/Shakti/NeuralNetworks")
endforeach()
