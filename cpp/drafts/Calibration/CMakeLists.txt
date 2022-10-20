add_library(
  DO_Sara_Calibration STATIC
  Chessboard.hpp Chessboard.cpp #
  HomographyDecomposition.cpp HomographyDecomposition.hpp #
  HomographyEstimation.cpp HomographyEstimation.hpp)
target_link_libraries(DO_Sara_Calibration PUBLIC DO::Sara::Core
                                                 DO::Sara::Graphics)
set_property(TARGET DO_Sara_Calibration PROPERTY FOLDER "Libraries/Sara")

add_subdirectory(tool)
