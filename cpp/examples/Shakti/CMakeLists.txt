if(CMAKE_CUDA_COMPILER)
  sara_message("Building DO::Shakti CUDA Libraries")
else()
  sara_message("Cannot use DO::Shakti CUDA libraries!")
endif()

find_package(
  DO_Sara
  COMPONENTS Core Graphics FeatureDescriptors ImageIO VideoIO
  REQUIRED)

if(DO_Sara_VideoIO_FOUND)
  # TODO: update code for CUDA 12.
  # add_subdirectory(ImageProcessing)
  add_subdirectory(Segmentation)
endif()

if(CMAKE_CUDA_COMPILER)
  add_subdirectory(FeatureDetectors)
  if(NvidiaVideoCodec_ROOT)
    add_subdirectory(VideoIO)
  endif()
endif()

add_subdirectory(OpenCL)
add_subdirectory(Vulkan)
