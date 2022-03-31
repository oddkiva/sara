if(NOT CMAKE_SYSTEM_NAME STREQUAL Emscripten)
  find_package(glew REQUIRED CONFIG)
  find_package(glfw3 REQUIRED)
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

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
    set_target_properties(
      ${filename}
      PROPERTIES
        LINK_FLAGS
        "-s ALLOW_MEMORY_GROWTH=1 -s USE_GLFW=3 -s USE_WEBGL2=1 -s FULL_ES3=1 -s USE_LIBJPEG=1 -s USE_LIBPNG=1 --preload-file ${CMAKE_CURRENT_BINARY_DIR}/assets/sunflowerField.jpg"
    )
  else()
    target_link_libraries(${filename} PRIVATE glfw)
    if(MSVC)
      target_link_libraries(${filename} PRIVATE GLEW::glew_s)
    endif()
  endif()
endmacro()

sara_add_wasm_executable(emscripten_hello_triangle.cpp)
sara_add_wasm_executable(emscripten_hello_texture.cpp)
