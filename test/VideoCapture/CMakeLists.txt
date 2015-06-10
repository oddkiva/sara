cmake_minimum_required(VERSION 2.6)

project(DO_VideoCapture_test)

message(STATUS "  - DO_VideoCapture_test")

find_package(DO_Sara COMPONENTS Core ImageProcessing Graphics REQUIRED)
include(${DO_USE_FILE})

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Multimedia REQUIRED)
find_package(Qt5MultimediaWidgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS}
                    ${Qt5Multimedia_INCLUDE_DIRS}
                    ${Qt5MultimediaWidgets_INCLUDE_DIRS})

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(SRCS main.cpp videosurface.cpp videowidget.cpp)
set(HDRS videosurface.hpp videowidget.hpp)
do_test(VideoCapture "${SRCS};${HDRS}" "${DO_LIBRARIES}" "VideoCapture")
qt5_use_modules(DO_VideoCapture_test Widgets Multimedia MultimediaWidgets)

set(CMAKE_AUTOMOC OFF)
set(CMAKE_INCLUDE_CURRENT_DIR OFF)
