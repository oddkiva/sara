find_package(GLFW3 REQUIRED)

file(GLOB SRC_FILES FILES *.cpp *.hpp *.cl)
source_group("OpenCL Source Files" REGULAR_EXPRESSION
  "${CMAKE_CURRENT_SOURCE_DIR}/.*\\.cl")

add_executable(hello_triangle ${SRC_FILES})
target_compile_definitions(hello_triangle PRIVATE -DGL_SILENCE_DEPRECATION)
target_link_libraries(hello_triangle PRIVATE
  ${DO_Sara_LIBRARIES}
  glfw)
target_link_libraries(hello_triangle PRIVATE DO_Sara_OpenCL_GL)
set_target_properties(hello_triangle PROPERTIES
  COMPILE_FLAGS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}")


if (MSVC)
  # Works but cumbersome.
  #install(FILES "${DIRS}/glew32.dll" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/\${CMAKE_INSTALL_CONFIG_NAME}")

  # Better.
  add_custom_command(TARGET hello_triangle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${GLEW_DLL}"
      "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/glew32.dll")
endif ()
