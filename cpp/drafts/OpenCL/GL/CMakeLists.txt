file(GLOB DO_Sara_OpenCL_GL_FILES FILES *.hpp *.cpp)

add_library(DO_Sara_OpenCL_GL ${DO_Sara_OpenCL_GL_FILES})

target_include_directories(DO_Sara_OpenCL_GL
  PUBLIC
  ${CMAKE_SOURCE_DIR}/cpp
  ${GLEW_INCLUDE_DIR})

target_compile_definitions(DO_Sara_OpenCL_GL
  PUBLIC
  GL_SILENCE_DEPRECATION)

target_link_libraries(DO_Sara_OpenCL_GL
  PRIVATE
  DO::Sara::Core

  PUBLIC
  $<$<STREQUAL:PLATFORM_ID,Emscripten>:OpenCL::OpenCL>
  ${OPENGL_LIBRARIES})

if (NOT APPLE)
  target_link_libraries(DO_Sara_OpenCL_GL
    PUBLIC
    ${GLEW_LIBRARIES})
endif ()

set_target_properties(DO_Sara_OpenCL_GL
  PROPERTIES
  FOLDER "Libraries/Sara")
