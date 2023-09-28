if (NOT SARA_BUILD_VIDEOIO)
  return ()
endif ()

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
                                                DO::Sara::VideoIO
                                                Boost::filesystem)
target_compile_definitions(ImageOrVideoReader PRIVATE -DBOOST_ALL_DYN_LINK #
                                                      -DBOOST_ALL_NO_LIB)
set_property(TARGET ImageOrVideoReader PROPERTY FOLDER
             "Examples/Sara/ChessboardDetection")

sara_add_example(find_chessboard_corners_2)
target_link_libraries(
  find_chessboard_corners_2
  PRIVATE DO::Sara::Graphics #
          DO::Sara::Geometry #
          ImageOrVideoReader #
          DO_Sara_ChessboardDetection)

sara_add_example(find_chessboard_corners)
target_link_libraries(
  find_chessboard_corners
  PRIVATE DO::Sara::Graphics #
          ImageOrVideoReader #
          DO_Sara_ChessboardDetection)

sara_add_example(detect_corners)
target_link_libraries(
  detect_corners
  PRIVATE DO::Sara::Graphics #
          ImageOrVideoReader #
          DO_Sara_ChessboardDetection)
