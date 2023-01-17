find_package(Qt${QT_VERSION} COMPONENTS Test REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(test_graphics_common STATIC
  event_scheduler.hpp event_scheduler.cpp)
target_link_libraries(test_graphics_common
                      PUBLIC Qt${QT_VERSION}::Widgets)
set_property(TARGET test_graphics_common
             PROPERTY FOLDER "Tests/Sara/Graphics")

set(CMAKE_AUTOMOC OFF)
set(CMAKE_INCLUDE_CURRENT_DIR OFF)


file(GLOB test_graphics_SOURCE_FILES FILES test_graphics_*.cpp)
foreach (file ${test_graphics_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(NAME ${filename}
                SOURCES ${file}
                DEPENDENCIES DO::Sara::Graphics
                             Qt${QT_VERSION}::Test
                             test_graphics_common
                FOLDER Graphics)
  target_compile_definitions(${filename}
    PRIVATE ${SARA_DEFINITIONS})
endforeach ()
