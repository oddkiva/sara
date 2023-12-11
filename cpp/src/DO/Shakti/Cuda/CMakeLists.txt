if (NOT CMAKE_CUDA_COMPILER)
  return ()
endif ()

sara_message("Building DO::Shakti CUDA Libraries")

add_subdirectory(Utilities)
add_subdirectory(MultiArray)
# FIXME: update the code for CUDA 12.
# add_subdirectory(ImageProcessing)
add_subdirectory(Segmentation)

if (NvidiaVideoCodec_ROOT)
  add_subdirectory(VideoIO)
endif ()

add_subdirectory(FeatureDetectors)

add_subdirectory(TensorRT)
