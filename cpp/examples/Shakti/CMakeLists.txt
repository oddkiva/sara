find_package(DO_Sara COMPONENTS
  Core Graphics FeatureDescriptors ImageIO VideoIO)

if (DO_Sara_VideoIO_FOUND)
  add_subdirectory(ImageProcessing)
  add_subdirectory(Segmentation)
endif ()
