set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(image_annotator main.cpp MainWindow.hpp MainWindow.cpp)
target_link_libraries(image_annotator
  PRIVATE
  Qt5::Gui
  Qt5::Widgets
  Qt5::OpenGL
  $<$<PLATFORM_ID:Windows>:Qt5::WinMain>)

set(CMAKE_AUTOMOC OFF)
set(CMAKE_INCLUDE_CURRENT_DIR OFF)
