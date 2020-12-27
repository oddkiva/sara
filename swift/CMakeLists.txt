cmake_minimum_required(VERSION 3.16)

project(Sara-Swift LANGUAGES CXX Swift)

add_subdirectory(Core)
add_subdirectory(Graphics)

add_executable(hello_world
  main.swift
  Graphics.swift
  Image.swift
  ImageIO.swift)
target_link_libraries(hello_world PRIVATE SaraSwiftCore SaraSwiftGraphics)
