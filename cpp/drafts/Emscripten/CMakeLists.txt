if(NOT CMAKE_SYSTEM_NAME STREQUAL Emscripten)
  find_package(glew REQUIRED)
  find_package(glfw3 REQUIRED)
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

list(
  APPEND
  GL_UTILS
  Geometry.hpp
  # Implementation V1
  MyGLFW.hpp
  MyGLFW.cpp
  Scene.hpp
  Scene.cpp
  # Implementation V2
  ImagePlaneRenderer.hpp
  ImagePlaneRenderer.cpp
  ImageDewarpRenderer.hpp
  ImageDewarpRenderer.cpp
  LineRenderer.hpp
  LineRenderer.cpp
  MetricGridRenderer.hpp
  MetricGridRenderer.cpp #
)
add_library(emscripten_gl_utilities ${GL_UTILS})
target_include_directories(emscripten_gl_utilities
                           PRIVATE ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(
  emscripten_gl_utilities
  PRIVATE DO_Sara_OpenCL_GL #
          DO::Sara::Core #
          DO::Sara::ImageIO
  INTERFACE $<$<NOT:$<PLATFORM_ID:Emscripten>>:glfw>)

if(MSVC)
  target_link_libraries(emscripten_gl_utilities INTERFACE GLEW::glew_s)
endif()
target_link_options(
  emscripten_gl_utilities #
  INTERFACE
  "$<$<PLATFORM_ID:Emscripten>:SHELL:-s USE_GLFW=3>" #
  "$<$<PLATFORM_ID:Emscripten>:SHELL:-s USE_WEBGL2=1>" #
  "$<$<PLATFORM_ID:Emscripten>:SHELL:-s FULL_ES3=1>")

macro(sara_add_wasm_executable file)
  get_filename_component(filename "${file}" NAME_WE)

  add_executable(${filename} ${file})
  set_target_properties(${filename} PROPERTIES SUFFIX ".html")

  target_include_directories(${filename} PRIVATE ${CMAKE_SOURCE_DIR}/cpp)
  target_link_libraries(
    ${filename}
    PRIVATE DO_Sara_OpenCL_GL #
            DO::Sara::Core #
            DO::Sara::ImageIO #
            $<$<NOT:$<PLATFORM_ID:Emscripten>>:glfw>)
  if(MSVC)
    target_link_libraries(${filename} PRIVATE GLEW::glew_s)
  endif()
endmacro()

set(ALL_FLAGS "SHELL:-s ALLOW_MEMORY_GROWTH=1")
set(ALL_FLAGS
    ${ALL_FLAGS}
    "SHELL:--preload-file ${CMAKE_CURRENT_BINARY_DIR}/assets/image-pinhole.png")
set(ALL_FLAGS
    ${ALL_FLAGS}
    "SHELL:--preload-file ${CMAKE_CURRENT_BINARY_DIR}/assets/image-omni.png")

# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------
sara_add_wasm_executable(test_image_plane_renderer.cpp)
target_link_libraries(test_image_plane_renderer #
                      PRIVATE emscripten_gl_utilities)
target_link_options(test_image_plane_renderer PRIVATE ${ALL_FLAGS})

sara_add_wasm_executable(test_line_renderer.cpp)
target_link_libraries(test_line_renderer #
                      PRIVATE emscripten_gl_utilities)
target_link_options(test_line_renderer PRIVATE ${ALL_FLAGS})

sara_add_wasm_executable(test_metric_grid_renderer.cpp)
target_link_libraries(test_metric_grid_renderer #
                      PRIVATE emscripten_gl_utilities ImGui)
target_link_options(test_metric_grid_renderer PRIVATE ${ALL_FLAGS})

sara_add_wasm_executable(test_image_dewarp_renderer.cpp)
target_link_libraries(test_image_dewarp_renderer #
                      PRIVATE emscripten_gl_utilities)
target_link_options(test_image_dewarp_renderer PRIVATE ${ALL_FLAGS})
