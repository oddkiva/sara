find_package(Qt5 COMPONENTS Core Widgets OpenGL REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (Qt5_POSITION_INDEPENDENT_CODE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

add_executable(ShaderExample main.cpp
  OpenGLWindow.cpp OpenGLWindow.hpp
  TriangleWindow.cpp TriangleWindow.hpp)
target_link_libraries(ShaderExample Qt5::Widgets Qt5::OpenGL)

set(CMAKE_AUTOMOC OFF)
