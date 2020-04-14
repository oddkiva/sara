file(GLOB SRC_FILES FILES *.cpp *.hpp *.cl)
source_group("OpenCL Source Files" REGULAR_EXPRESSION
  "${CMAKE_CURRENT_SOURCE_DIR}/.*\\.cl")

add_executable(image_processing ${SRC_FILES})

target_link_libraries(image_processing
  PRIVATE
  ${DO_Sara_LIBRARIES}
  DO_Sara_OpenCL_Core
  DO_Sara_OpenCL_GL
  glfw)

target_compile_definitions(image_processing
  PRIVATE
  SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}
  $<$<PLATFORM_ID:Darwin>:GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED>
  $<$<PLATFORM_ID:Darwin>:CL_SILENCE_DEPRECATION>)

set_target_properties(image_processing
  PROPERTIES
  FOLDER "Tutorials/OpenCL")

if (MSVC)
  add_custom_command(TARGET image_processing POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${GLEW_DLL}"
      "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/glew32.dll")
endif ()
