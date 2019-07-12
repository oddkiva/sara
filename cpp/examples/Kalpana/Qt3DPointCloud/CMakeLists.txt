find_package(Qt5 COMPONENTS Core Widgets 3DCore 3DExtras 3DRender 3DInput REQUIRED)


set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (Qt5_POSITION_INDEPENDENT_CODE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

add_executable(Qt3DPointCloud main.cpp)
target_link_libraries(Qt3DPointCloud
  Qt5::Widgets Qt5::3DCore Qt5::3DExtras Qt5::3DInput)

set(CMAKE_AUTOMOC OFF)
