find_package(DO_Sara
  COMPONENTS
  Core Graphics ImageIO VideoIO                           # Classic stuff
  DisjointSets FeatureDetectors Geometry ImageProcessing  # Algorithmic modules
  Visualization
  REQUIRED)

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

sara_add_example(curvature_example)
sara_add_example(fast_marching_example)
sara_add_example(level_set_2d_example)

sara_add_example(optical_flow_example)

sara_add_example(watershed_example)
sara_add_example(watershed_v2_example)
target_link_libraries(watershed_v2_example
  PRIVATE
  $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)

sara_add_example(edge_detection_example)
target_include_directories(edge_detection_example
  PRIVATE ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(edge_detection_example
  PRIVATE
  Boost::filesystem
  $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)
target_compile_definitions(edge_detection_example
  PRIVATE
  -DBOOST_ALL_DYN_LINK
  -DBOOST_ALL_NO_LIB)

sara_add_example(line_segment_detection_example)
target_link_libraries(line_segment_detection_example
  PRIVATE $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)


add_library(ChessboardDetection STATIC
  Chessboard/Corner.hpp
  Chessboard/Corner.cpp
  Chessboard/CircularProfileExtractor.hpp
  Chessboard/CircularProfileExtractor.cpp
  Chessboard/Erode.hpp
  Chessboard/JunctionDetection.hpp
  Chessboard/JunctionDetection.cpp
  Chessboard/NonMaximumSuppression.hpp
  Chessboard/SquareReconstruction.hpp
  Chessboard/SquareReconstruction.cpp)
set_target_properties(ChessboardDetection
  PROPERTIES
  FOLDER "Libraries")
target_link_libraries(ChessboardDetection PUBLIC ${DO_Sara_LIBRARIES})

sara_add_example(find_chessboard_corners)
target_link_libraries(find_chessboard_corners PRIVATE ChessboardDetection)

sara_add_example(find_chessboard_corners_2)
target_link_libraries(find_chessboard_corners_2 PRIVATE ChessboardDetection)

sara_add_example(find_chessboard_corners_3)
target_link_libraries(find_chessboard_corners_3
  PRIVATE
  ChessboardDetection
  Boost::filesystem)

sara_add_example(find_aruco_marker)
if (TBB_FOUND)
  target_link_libraries(find_aruco_marker PRIVATE TBB::tbb)
endif ()
