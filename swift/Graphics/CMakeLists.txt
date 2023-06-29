find_package(DO_Sara COMPONENTS Graphics ImageIO VideoIO REQUIRED)

add_library(SaraSwiftGraphics
  CGraphics.hpp CGraphics.cpp
  Graphics.hpp Graphics.cpp)
target_include_directories(SaraSwiftGraphics
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}  # DO NOT REMOVE!
  PRIVATE
  ${JPEG_INCLUDE_DIR}
  ${PNG_INCLUDE_DIRS}
  ${TIFF_INCLUDE_DIR}
  ${ZLIB_INCLUDE_DIRS})
target_link_libraries(SaraSwiftGraphics
  PRIVATE
  ${DO_Sara_LIBRARIES})
set_target_properties(SaraSwiftGraphics PROPERTIES
  FOLDER "Swift")
