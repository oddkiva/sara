add_executable(glfw_html main.cpp)

if(CMAKE_SYSTEM_NAME STREQUAL Emscripten)
  set_target_properties(glfw_html PROPERTIES SUFFIX ".html")

  target_include_directories(glfw_html PRIVATE ${CMAKE_SOURCE_DIR}/cpp)
  target_compile_definitions(glfw_html PRIVATE -DEMSCRIPTEN=1)
  target_link_libraries(glfw_html PRIVATE DO_Sara_OpenCL_GL DO::Sara::Core)
  set_target_properties(
    glfw_html PROPERTIES LINK_FLAGS "-s USE_GLFW=3 -s USE_WEBGL2=1 -s FULL_ES3=1")
else()
  find_package(glfw3 REQUIRED)
  target_link_libraries(glfw_html PRIVATE glfw)
endif()

if(MSVC)
  find_package(glew REQUIRED CONFIG)
  target_link_libraries(glfw_html PRIVATE GLEW::glew_s)
endif()
