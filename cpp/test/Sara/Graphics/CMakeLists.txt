find_package(DO_Sara COMPONENTS Core Graphics REQUIRED)
find_package(Qt5 COMPONENTS Test REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(test_graphics_common STATIC
  event_scheduler.hpp event_scheduler.cpp)
target_link_libraries(test_graphics_common PUBLIC Qt5::Widgets)
set_property(TARGET test_graphics_common
             PROPERTY FOLDER "Tests/Sara/Graphics")

set(CMAKE_AUTOMOC OFF)
set(CMAKE_INCLUDE_CURRENT_DIR OFF)


file(GLOB test_graphics_SOURCE_FILES FILES test_graphics_*.cpp)
foreach (file ${test_graphics_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    ${filename} ${file}
    "${DO_Sara_LIBRARIES};Qt5::Test;test_graphics_common"
    "Graphics")
  set_target_properties(${filename} PROPERTIES
                        COMPILE_FLAGS ${SARA_DEFINITIONS})
endforeach ()
