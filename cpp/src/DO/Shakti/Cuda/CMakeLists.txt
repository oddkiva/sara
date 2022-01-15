add_subdirectory(Utilities)
add_subdirectory(MultiArray)
add_subdirectory(ImageProcessing)

if (NvidiaVideoCodec_ROOT)
  add_subdirectory(VideoIO)
endif ()

# add_subdirectory(FeatureDetectors)
