find_package(OpenMP)
find_package(DO_Sara COMPONENTS Core DisjointSets Graphics ImageIO
  ImageProcessing VideoIO REQUIRED)

macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} PRIVATE ${DO_Sara_LIBRARIES})
  set_property(
    TARGET ${example} PROPERTY
    FOLDER "Examples/Sara/ImageProcessing")
endmacro()

sara_add_example(image_processing_example)
sara_add_example(infinite_image_example)
sara_add_example(gemm_based_convolution_example)
sara_add_example(watershed_example)

sara_add_example(edge_detection_example)
target_link_libraries(edge_detection_example
  PRIVATE $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)

sara_add_example(line_segment_detection_example)
target_link_libraries(line_segment_detection_example
  PRIVATE $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)

sara_add_example(birds_eye_view_example)
