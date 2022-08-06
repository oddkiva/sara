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
  ChessboardDetector.cpp)
set_target_properties(DO_Sara_ChessboardDetection PROPERTIES FOLDER "Libraries")
target_link_libraries(
  DO_Sara_ChessboardDetection
  PUBLIC DO::Sara::ImageProcessing #
         DO::Sara::FeatureDetectors
  PRIVATE Boost::filesystem)
target_compile_definitions(
  DO_Sara_ChessboardDetection #
  PRIVATE -DBOOST_ALL_DYN_LINK #
          -DBOOST_ALL_NO_LIB)

add_subdirectory(examples)
