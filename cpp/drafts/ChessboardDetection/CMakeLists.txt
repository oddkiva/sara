find_package(
  Boost
  COMPONENTS log
  REQUIRED)

add_library(
  DO_Sara_ChessboardDetection STATIC
  # Feature detection.
  Corner.hpp
  Corner.cpp
  CircularProfileExtractor.hpp
  CircularProfileExtractor.cpp
  Erode.hpp
  EdgeStatistics.hpp
  EdgeStatistics.cpp
  JunctionDetection.hpp
  JunctionDetection.cpp
  NonMaximumSuppression.hpp
  # Edge parsing.
  LineReconstruction.hpp
  LineReconstruction.cpp
  SquareReconstruction.hpp
  SquareReconstruction.cpp
  # Experimental
  EllipseFitting.hpp
  EllipseFitting.cpp
  # The class that wraps everything.
  ChessboardDetector.hpp
  ChessboardDetector.cpp
  ChessboardDetectorV2.hpp
  ChessboardDetectorV2.cpp)
set_target_properties(DO_Sara_ChessboardDetection PROPERTIES FOLDER
                                                             "Libraries/Sara")
target_link_libraries(
  DO_Sara_ChessboardDetection
  PUBLIC DO::Sara::ImageProcessing #
         DO::Sara::FeatureDetectors
  PRIVATE Boost::log)

add_subdirectory(examples)
