project (easyexif)

cmake_minimum_required(VERSION 2.8)

add_library (easyexif exif.h exif.cpp)
install(TARGETS easyexif
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        COMPONENT ThirdParty)
