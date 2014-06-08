cmake_minimum_required(VERSION 2.6)

project(DO_Graphics_test)

if (DO_DISABLE_SPECIFIC_TEST_SUITES_FOR_TRAVIS_CI)
  message("Warning: some test suites are disabled for Travis CI.")
  add_definitions("-DDISABLE_IN_TRAVIS_CI")
endif ()

message(STATUS "  - DO_Graphics_test")

find_package(DO COMPONENTS Core Graphics REQUIRED)
include(${DO_USE_FILE})

find_package(Qt5Test REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${Qt5Test_INCLUDE_DIRS}
                    ${DO_DIR}/build/test/Graphics)

set(TestGroupName "Graphics")
do_unit_test(test_painting_window test_painting_window.cpp
             "${DO_LIBRARIES};${Qt5Test_LIBRARIES}" "${TestGroupName}")
do_unit_test(test_opengl_window test_opengl_window.cpp
             "${DO_LIBRARIES};${Qt5Test_LIBRARIES}" "${TestGroupName}")
do_unit_test(test_graphics_view test_graphics_view.cpp
             "${DO_LIBRARIES};${Qt5Test_LIBRARIES}" "${TestGroupName}")

set(CMAKE_AUTOMOC OFF)
set(CMAKE_INCLUDE_CURRENT_DIR OFF)
