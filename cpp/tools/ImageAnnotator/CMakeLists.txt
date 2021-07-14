set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(image_annotator main.cpp MainWindow.hpp MainWindow.cpp)
target_link_libraries(image_annotator
  PRIVATE
  Qt${QT_VERSION}::Gui
  Qt${QT_VERSION}::Widgets
  Qt${QT_VERSION}::OpenGL
  $<$<PLATFORM_ID:Windows>:Qt${QT_VERSION}::WinMain>)
set_property(TARGET image_annotator
  PROPERTY
  FOLDER "Tools")

set(CMAKE_AUTOMOC OFF)
set(CMAKE_INCLUDE_CURRENT_DIR OFF)
