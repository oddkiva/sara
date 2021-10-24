find_package(DO_Sara COMPONENTS Graphics ImageIO VideoIO REQUIRED)

add_library(SaraSwiftImageProcessing CImageProcessing.hpp CImageProcessing.cpp)
target_include_directories(SaraSwiftImageProcessing
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SaraSwiftImageProcessing
  PRIVATE
  ${DO_Sara_LIBRARIES}
  Halide::Halide
  Halide::Runtime
  shakti_rgb8u_to_gray32f_cpu
  shakti_gray32f_to_rgb8u_cpu)
set_target_properties(SaraSwiftImageProcessing PROPERTIES
  FOLDER "Swift")
