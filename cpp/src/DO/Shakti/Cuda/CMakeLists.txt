add_subdirectory(Utilities)
add_subdirectory(MultiArray)
add_subdirectory(ImageProcessing)
add_subdirectory(Segmentation)

if (NvidiaVideoCodec_ROOT)
  add_subdirectory(VideoIO)
endif ()

# add_subdirectory(FeatureDetectors)
