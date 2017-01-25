project (easyexif)

cmake_minimum_required(VERSION 2.8)

add_library (easyexif STATIC exif.h exif.cpp)
set_target_properties(easyexif PROPERTIES OUTPUT_NAME_DEBUG easyexif-d)

configure_file(${DO_Sara_DIR}/cmake/FindEasyEXIF.cmake.in
               ${CMAKE_BINARY_DIR}/cmake/FindEasyEXIF.cmake @ONLY)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  install(DIRECTORY ${DO_Sara_ThirdParty_DIR}/easyexif
          DESTINATION include
          COMPONENT ThirdParty)
endif ()
install(
  TARGETS easyexif
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  COMPONENT ThirdParty)
