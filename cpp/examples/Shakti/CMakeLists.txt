find_package(DO_Sara COMPONENTS
  Core Graphics FeatureDescriptors ImageIO VideoIO REQUIRED)

if (DO_Sara_VideoIO_FOUND)
  add_subdirectory(ImageProcessing)
  add_subdirectory(Segmentation)
endif ()

if (CMAKE_CUDA_COMPILER)
  add_subdirectory(FeatureDetectors)
  if (NvidiaVideoCodec_ROOT)
    add_subdirectory(VideoIO)
  endif ()
endif ()
