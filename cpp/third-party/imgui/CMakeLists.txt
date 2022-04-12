project(ImGui)

cmake_minimum_required(VERSION 3.12)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Emscripten")
  set(USE_FLAGS
      "SHELL:-s USE_GLFW=3"
      "SHELL:-s FULL_ES3=1"
      "SHELL:-s USE_FREETYPE=1"
      "SHELL:-s ALLOW_MEMORY_GROWTH=1"
      "SHELL:-s NO_EXIT_RUNTIME=0"
      "SHELL:-s ASSERTIONS=1"
      "SHELL:-s DISABLE_EXCEPTION_CATCHING=1")
else()
  find_package(SDL2 REQUIRED)
  find_package(Freetype REQUIRED)
endif()

add_library(
  ImGui
  # FILES
  backends/imgui_impl_glfw.h
  backends/imgui_impl_glfw.cpp

  backends/imgui_impl_opengl3.h
  backends/imgui_impl_opengl3.cpp

  imconfig.h
  imgui_internal.h
  imgui.h
  imgui.cpp
  imgui_draw.cpp
  imgui_tables.cpp
  imgui_widgets.cpp)
target_include_directories(ImGui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                        ${CMAKE_CURRENT_SOURCE_DIR}/backends)

add_library(ImGui_demo imgui_demo.cpp)
target_link_libraries(ImGui_demo PUBLIC ImGui)

# add_executable(imgui_example examples/example_emscripten_opengl3/main.cpp)
# target_link_libraries(imgui_example PRIVATE ImGui ImGui_demo
# ${SDL2_LIBRARIES})
