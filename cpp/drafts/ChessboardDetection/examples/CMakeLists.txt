macro(sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} PRIVATE ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER
                                          "Examples/Sara/ChessboardDetection")
endmacro()

add_library(ImageOrVideoReader STATIC Utilities/ImageOrVideoReader.hpp
                                      Utilities/ImageOrVideoReader.cpp)
target_link_libraries(ImageOrVideoReader PUBLIC DO::Sara::ImageIO
                                                DO::Sara::VideoIO)

sara_add_example(find_chessboard_corners)
target_link_libraries(
  find_chessboard_corners #
  PRIVATE DO::Sara::Graphics #
          ImageOrVideoReader #
          DO_Sara_ChessboardDetection)

sara_add_example(find_chessboard_corners_2)
target_link_libraries(
  find_chessboard_corners_2
  PRIVATE DO::Sara::Graphics #
          DO::Sara::Geometry #
          ImageOrVideoReader #
          DO_Sara_ChessboardDetection)

sara_add_example(find_chessboard_corners_3)
target_link_libraries(
  find_chessboard_corners_3
  PRIVATE DO::Sara::Graphics #
          ImageOrVideoReader #
          DO_Sara_ChessboardDetection)

sara_add_example(local_radon_transform_for_corner_detection)
target_link_libraries(
  local_radon_transform_for_corner_detection
  PRIVATE DO::Sara::Graphics #
          ImageOrVideoReader #
          DO_Sara_ChessboardDetection)
