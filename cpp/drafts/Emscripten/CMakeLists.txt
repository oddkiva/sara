if(NOT CMAKE_SYSTEM_NAME STREQUAL Emscripten)
  find_package(glew REQUIRED CONFIG)
  find_package(glfw3 REQUIRED)
else()
  set(BASE_LINK_FLAGS "-s USE_GLFW=3 -s USE_WEBGL2=1 -s FULL_ES3=1")
  set(SARA_IMAGEIO_LINK_FLAGS "-s USE_LIBJPEG=1 -s USE_LIBPNG=1")
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

macro(sara_add_wasm_library library_name files)
  if(CMAKE_SYSTEM_NAME STREQUAL Emscripten)
    add_library(${library_name} ${files})

    target_include_directories(${library_name} PRIVATE ${CMAKE_SOURCE_DIR}/cpp)
    target_compile_options(${library_name} PUBLIC "SHELL:-O3")
    target_link_libraries(
      ${library_name}
      PRIVATE DO_Sara_OpenCL_GL #
              DO::Sara::Core #
              DO::Sara::ImageIO)
  else()
    target_link_libraries(${library_name} PRIVATE glfw)
    if(MSVC)
      target_link_libraries(${library_name} PRIVATE GLEW::glew_s)
    endif()
  endif()
endmacro()

macro(sara_add_wasm_executable file)
  get_filename_component(filename "${file}" NAME_WE)

  if(CMAKE_SYSTEM_NAME STREQUAL Emscripten)
    add_executable(${filename} ${file})
    set_target_properties(${filename} PROPERTIES SUFFIX ".html")

    target_include_directories(${filename} PRIVATE ${CMAKE_SOURCE_DIR}/cpp)
    target_compile_options(${filename} PUBLIC "SHELL:-O3")
    target_link_libraries(
      ${filename}
      PRIVATE DO_Sara_OpenCL_GL #
              DO::Sara::Core #
              DO::Sara::ImageIO)
  else()
    target_link_libraries(${filename} PRIVATE glfw)
    if(MSVC)
      target_link_libraries(${filename} PRIVATE GLEW::glew_s)
    endif()
  endif()
endmacro()

list(
  APPEND
  GL_UTILS
  Geometry.hpp
  MyGLFW.hpp
  MyGLFW.cpp
  Scene.hpp
  Scene.cpp
  LinePainter.hpp
  LinePainter.cpp
  MetricGridRenderer.hpp
  MetricGridRenderer.cpp #
)
sara_add_wasm_library(emscripten_gl_utilities "${GL_UTILS}")

sara_add_wasm_executable(emscripten_hello_texture.cpp)
target_link_libraries(emscripten_hello_texture PRIVATE emscripten_gl_utilities)

set(ALL_FLAGS
    "${BASE_LINK_FLAGS} ${SARA_IMAGEIO_LINK_FLAGS} -s ALLOW_MEMORY_GROWTH=1")
set(ALL_FLAGS
    "${ALL_FLAGS} --preload-file ${CMAKE_CURRENT_BINARY_DIR}/assets/image-pinhole.png"
)
set(ALL_FLAGS
    "${ALL_FLAGS} --preload-file ${CMAKE_CURRENT_BINARY_DIR}/assets/image-omni.png"
)
set_target_properties(emscripten_hello_texture #
                      PROPERTIES LINK_FLAGS ${ALL_FLAGS})
