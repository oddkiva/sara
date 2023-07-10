if(NOT CMAKE_SYSTEM_NAME STREQUAL Emscripten AND NOT GLEW_FOUND)
  return()
endif()

set(SARA_EMSCRIPTEN_BUILD_HTML NO)

if(NOT CMAKE_SYSTEM_NAME STREQUAL Emscripten)
  add_custom_target(emscripten_copy_assets)
  add_custom_command(
    TARGET emscripten_copy_assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/bin/assets)
else()
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets
       DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

list(
  APPEND
  GL_UTILS
  Geometry.hpp
  ImagePlaneRenderer.hpp
  ImagePlaneRenderer.cpp
  ImageDewarpRenderer.hpp
  ImageDewarpRenderer.cpp
  LineRenderer.hpp
  LineRenderer.cpp
  MetricGridRenderer.hpp
  MetricGridRenderer.cpp #
)
add_library(emscripten_gl_utilities STATIC ${GL_UTILS})
target_include_directories(emscripten_gl_utilities
                           PRIVATE ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(
  emscripten_gl_utilities
  PRIVATE DO::Sara::Core #
          DO::Sara::ImageIO #
          DO::Kalpana::EasyGL #
  INTERFACE $<$<NOT:$<PLATFORM_ID:Emscripten>>:Boost::filesystem>
            $<$<NOT:$<PLATFORM_ID:Emscripten>>:glfw>)
set_property(TARGET emscripten_gl_utilities
             PROPERTY FOLDER "Libraries/Sara/Emscripten")

if(MSVC)
  target_link_libraries(emscripten_gl_utilities INTERFACE GLEW::GLEW)
endif()
target_link_options(
  emscripten_gl_utilities #
  INTERFACE
  "$<$<PLATFORM_ID:Emscripten>:SHELL:-s USE_GLFW=3>" #
  "$<$<PLATFORM_ID:Emscripten>:SHELL:-s USE_WEBGL2=1>" #
  "$<$<PLATFORM_ID:Emscripten>:SHELL:-s FULL_ES3=1>"
  "$<$<PLATFORM_ID:Emscripten>:SHELL:-s EXPORT_ES6=1>")

if(NOT SARA_EMSCRIPTEN_BUILD_HTML)
  target_link_options(
    emscripten_gl_utilities
    INTERFACE
    "$<$<PLATFORM_ID:Emscripten>:SHELL:-s MODULARIZE=1>"
    "$<$<PLATFORM_ID:Emscripten>:SHELL:-s ENVIRONMENT='web'>"
    "$<$<PLATFORM_ID:Emscripten>:SHELL:-s EXPORTED_RUNTIME_METHODS=\"['specialHTMLTargets', 'JSEvents', 'GL', 'callMain', 'abort']\">"
  )
endif()

macro(sara_add_wasm_executable file)
  get_filename_component(filename "${file}" NAME_WE)

  add_executable(${filename} ${file})
  if(CMAKE_SYSTEM_NAME STREQUAL Emscripten AND SARA_EMSCRIPTEN_BUILD_HTML)
    set_target_properties(${filename} PROPERTIES SUFFIX ".html")
  endif()

  target_include_directories(${filename} PRIVATE ${CMAKE_SOURCE_DIR}/cpp)
  target_link_libraries(
    ${filename}
    PRIVATE DO::Sara::Core #
            DO::Sara::ImageIO #
            DO::Kalpana::EasyGL #
            $<$<NOT:$<PLATFORM_ID:Emscripten>>:glfw>)
  if(MSVC)
    target_link_libraries(${filename} PRIVATE GLEW::GLEW)
  endif()

  set_property(TARGET ${filename} PROPERTY FOLDER "Examples/Sara/Emscripten")

  if(NOT CMAKE_SYSTEM_NAME STREQUAL Emscripten)
    add_dependencies(emscripten_copy_assets ${filename})
    set_property(TARGET emscripten_copy_assets
                 PROPERTY FOLDER "Examples/Sara/Emscripten")
  endif()
endmacro()

set(ALL_FLAGS "$<$<PLATFORM_ID:Emscripten>:SHELL:-s ALLOW_MEMORY_GROWTH=1>")
set(ALL_FLAGS
    ${ALL_FLAGS}
    "$<$<PLATFORM_ID:Emscripten>:SHELL:--preload-file ${CMAKE_CURRENT_BINARY_DIR}/assets/image-pinhole.png>"
)
set(ALL_FLAGS
    ${ALL_FLAGS}
    "$<$<PLATFORM_ID:Emscripten>:SHELL:--preload-file ${CMAKE_CURRENT_BINARY_DIR}/assets/image-omni.png>"
)

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
target_link_options(
  test_image_dewarp_renderer PRIVATE
  "$<$<PLATFORM_ID:Emscripten>:SHELL:-s ALLOW_MEMORY_GROWTH=1>")

target_link_options(
  test_metric_grid_renderer
  PRIVATE
  "$<$<PLATFORM_ID:Emscripten>:SHELL:--shell-file ${CMAKE_CURRENT_SOURCE_DIR}/http/index.html>"
)