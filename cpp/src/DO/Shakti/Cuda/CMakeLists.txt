# OK WORKS
add_subdirectory(Utilities)
add_subdirectory(MultiArray)

# target_link_libraries(DO_Shakti_ImageProcessing
#   PUBLIC
#   DO_Shakti_MultiArray
#   DO_Shakti_Utilities)
# target_link_libraries(DO_Shakti_Segmentation
#   PUBLIC
#   DO_Shakti_MultiArray
#   DO_Shakti_Utilities)

if (NvidiaVideoCodec_ROOT)
  add_subdirectory(VideoIO)
endif ()

# add_subdirectory(FeatureDetectors)
