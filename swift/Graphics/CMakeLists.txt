if(UNIX
   AND NOT APPLE
   AND NOT SWIFT_BRIDGING_INCLUDE_DIR)
  message(FATAL_ERROR "SWIFT_BRIDGING_INCLUDE_DIR must be defined!")
endif()

add_library(
  SaraSwiftGraphics
  Graphics.hpp Graphics.cpp #
  ImageIO.hpp ImageIO.cpp #
  VideoIO.hpp VideoIO.cpp)
target_include_directories(
  SaraSwiftGraphics
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} # DO NOT REMOVE!
         $<$<PLATFORM_ID:Linux>:${SWIFT_BRIDGING_INCLUDE_DIR}>
  PRIVATE ${JPEG_INCLUDE_DIR} ${PNG_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR}
          ${ZLIB_INCLUDE_DIRS})
target_link_libraries(
  SaraSwiftGraphics
  PRIVATE DO::Sara::Graphics #
          DO::Sara::ImageIO #
          DO::Sara::VideoIO)
set_target_properties(
  SaraSwiftGraphics #
  PROPERTIES FOLDER "Swift" #
             Swift_MODULE_NAME "SaraGraphics")
target_compile_options(
  SaraSwiftGraphics
  PUBLIC "$<$<COMPILE_LANGUAGE:Swift>:-cxx-interoperability-mode=default>")
