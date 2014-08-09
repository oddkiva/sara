cmake_minimum_required(VERSION 2.6)

project(DO_Graphics_test)

message(STATUS "  - DO_Graphics_tests")

find_package(DO COMPONENTS Core Graphics REQUIRED)
include(${DO_USE_FILE})

find_package(Qt5Test REQUIRED)

include_directories(${Qt5Test_INCLUDE_DIRS})

set(TestGroupName "Graphics")

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
add_library(Graphics_Tests_Common
            event_scheduler.hpp event_scheduler.cpp)
set_property(TARGET Graphics_Tests_Common
             PROPERTY FOLDER "DO Tests/${TestGroupName}")
set(CMAKE_AUTOMOC OFF)
set(CMAKE_INCLUDE_CURRENT_DIR OFF)

do_test(Graphics_DerivedQObjects_PaintingWindow test_painting_window.cpp
        "${DO_LIBRARIES};${Qt5Test_LIBRARIES};Graphics_Tests_Common" ${TestGroupName})
do_test(Graphics_DerivedQObjects_OpenGLWindow test_opengl_window.cpp
        "${DO_LIBRARIES};${Qt5Test_LIBRARIES};Graphics_Tests_Common" ${TestGroupName})
do_test(Graphics_DerivedQObjects_GraphicsView test_graphics_view.cpp
        "${DO_LIBRARIES};${Qt5Test_LIBRARIES};Graphics_Tests_Common" ${TestGroupName})

do_test(Graphics_Window_Management test_window_management.cpp
        "${DO_LIBRARIES};Graphics_Tests_Common" ${TestGroupName})
do_test(Graphics_Keyboard_Mouse_Input test_keyboard_mouse_input.cpp
        "${DO_LIBRARIES};Graphics_Tests_Common" ${TestGroupName})
do_test(Graphics_Sleep_Functions test_sleep_functions.cpp
        "${DO_LIBRARIES};Graphics_Tests_Common" ${TestGroupName})
do_test(Graphics_Painting_Commands test_painting_commands.cpp
        "${DO_LIBRARIES};${Qt5Test_LIBRARIES};Graphics_Tests_Common" ${TestGroupName})