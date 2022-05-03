if (NOT SARA_USE_OPENCV)
  return()
endif ()

find_package(OpenCV)

add_library(
  HomographyUtilities HomographyDecomposition.cpp HomographyDecomposition.hpp
                      HomographyEstimation.cpp HomographyEstimation.hpp)
target_link_libraries(HomographyUtilities PUBLIC DO::Sara::Core ${OpenCV_LIBS})

add_subdirectory(tool)
