if(CMAKE_CUDA_COMPILER)
  sara_message("Building DO::Shakti CUDA Libraries")

  if(DO_Sara_VideoIO_FOUND)
    # TODO: update code for CUDA 12.
    # add_subdirectory(ImageProcessing)
    add_subdirectory(Segmentation)
  endif()
  add_subdirectory(FeatureDetectors)

  if(NvidiaVideoCodec_ROOT)
    add_subdirectory(VideoIO)
  endif()
endif()

add_subdirectory(OpenCL)
add_subdirectory(Vulkan)

add_subdirectory(TensorRT)
