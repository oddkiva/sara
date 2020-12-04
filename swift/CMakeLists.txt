cmake_minimum_required(VERSION 3.16)

project(Sara-Swift LANGUAGES CXX Swift)

find_package(DO_Sara COMPONENTS Core REQUIRED)

add_library(SaraSwiftCore CCore.cpp)
# This is to find module.modulemap related to CCore.hpp
target_include_directories(SaraSwiftCore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SaraSwiftCore PRIVATE ${DO_Sara_LIBRARIES})

add_executable(hello_world hello_world.swift)
target_link_libraries(hello_world PRIVATE SaraSwiftCore)
