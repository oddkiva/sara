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

add_library(ChessboardDetection STATIC
  # Utilities
  Utilities/ImageOrVideoReader.hpp
  Utilities/ImageOrVideoReader.cpp
  # Feature detection.
  Chessboard/Corner.hpp
  Chessboard/Corner.cpp
  Chessboard/CircularProfileExtractor.hpp
  Chessboard/CircularProfileExtractor.cpp
  Chessboard/Erode.hpp
  Chessboard/EdgeStatistics.hpp
  Chessboard/EdgeStatistics.cpp
  Chessboard/JunctionDetection.hpp
  Chessboard/JunctionDetection.cpp
  Chessboard/NonMaximumSuppression.hpp
  # Edge parsing.
  Chessboard/LineReconstruction.hpp
  Chessboard/LineReconstruction.cpp
  Chessboard/SquareReconstruction.hpp
  Chessboard/SquareReconstruction.cpp
  # The class that wraps everything.
  Chessboard/ChessboardDetector.hpp
  Chessboard/ChessboardDetector.cpp)
set_target_properties(ChessboardDetection
  PROPERTIES
  FOLDER "Libraries")
target_link_libraries(ChessboardDetection
  PUBLIC ${DO_Sara_LIBRARIES}
  PRIVATE Boost::filesystem)
target_compile_definitions(ChessboardDetection
  PRIVATE
  -DBOOST_ALL_DYN_LINK
  -DBOOST_ALL_NO_LIB)

sara_add_example(find_chessboard_corners)
target_link_libraries(find_chessboard_corners PRIVATE ChessboardDetection)

sara_add_example(find_chessboard_corners_2)
target_link_libraries(find_chessboard_corners_2 PRIVATE ChessboardDetection)

sara_add_example(find_chessboard_corners_3)
target_link_libraries(find_chessboard_corners_3 PRIVATE ChessboardDetection)

sara_add_example(local_radon_transform_for_corner_detection)
target_link_libraries(local_radon_transform_for_corner_detection PRIVATE ChessboardDetection)

sara_add_example(find_aruco_marker)
target_link_libraries(find_aruco_marker PRIVATE Boost::filesystem)
if (TBB_FOUND)
  target_link_libraries(find_aruco_marker PRIVATE TBB::tbb)
endif ()
