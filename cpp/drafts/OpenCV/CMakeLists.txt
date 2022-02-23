find_package(
  DO_Sara
  COMPONENTS Core Graphics ImageProcessing ImageIO VideoIO
  REQUIRED)
find_package(OpenCV REQUIRED)

add_library(
  HomographyUtilities HomographyDecomposition.cpp HomographyDecomposition.hpp
                      HomographyEstimation.cpp HomographyEstimation.hpp)
target_link_libraries(HomographyUtilities PUBLIC DO::Sara::Core ${OpenCV_LIBS})

add_subdirectory(tool)
