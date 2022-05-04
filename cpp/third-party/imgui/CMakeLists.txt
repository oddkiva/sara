project(ImGui)

cmake_minimum_required(VERSION 3.12)

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

target_link_libraries(ImGui PUBLIC $<$<BOOL:${FREETYPE_FOUND}>:Freetype::Freetype>)
target_include_directories(ImGui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                        ${CMAKE_CURRENT_SOURCE_DIR}/backends)

add_library(ImGui_demo imgui_demo.cpp)
target_link_libraries(ImGui_demo PUBLIC ImGui)
