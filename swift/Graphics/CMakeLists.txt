add_library(SaraSwiftGraphics CGraphics.hpp CGraphics.cpp Graphics.hpp
                              Graphics.cpp)
target_include_directories(
  SaraSwiftGraphics
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} # DO NOT REMOVE!
  PRIVATE ${JPEG_INCLUDE_DIR} ${PNG_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR}
          ${ZLIB_INCLUDE_DIRS})
target_link_libraries(
  SaraSwiftGraphics
  PRIVATE DO::Sara::Graphics #
          DO::Sara::ImageIO #
          DO::Sara::VideoIO)
set_target_properties(SaraSwiftGraphics PROPERTIES FOLDER "Swift")
set_target_properties(SaraSwiftGraphics PROPERTIES Swift_MODULE_NAME
                                                   "SaraGraphics")
target_compile_options(
  SaraSwiftGraphics
  PUBLIC "$<$<COMPILE_LANGUAGE:Swift>:-cxx-interoperability-mode=default>")
