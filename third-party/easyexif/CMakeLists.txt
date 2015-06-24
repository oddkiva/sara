project (easyexif)

cmake_minimum_required(VERSION 2.8)

add_library (easyexif exif.h exif.cpp)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  install(DIRECTORY ${DO_Sara_ThirdParty_DIR}/easyexif
          DESTINATION include
          COMPONENT ThirdParty)
  install(TARGETS easyexif
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
          COMPONENT ThirdParty)
endif ()