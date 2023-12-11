if(CMAKE_CUDA_COMPILER)
  sara_message("Building DO::Shakti Libraries")
else()
  sara_message("Cannot use DO::Shakti libraries!")
  return()
endif()

add_subdirectory(Utilities)
add_subdirectory(MultiArray)
# TODO: update code for CUDA 12. add_subdirectory(ImageProcessing)
add_subdirectory(Segmentation)
add_subdirectory(FeatureDetectors)

add_subdirectory(TensorRT)
