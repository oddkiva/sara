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

set_target_properties(image_processing
  PROPERTIES
  COMPILE_FLAGS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}"
  FOLDER "Tutorials/OpenCL")

if (MSVC)
  # Works but cumbersome.
  #install(FILES "${DIRS}/glew32.dll" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/\${CMAKE_INSTALL_CONFIG_NAME}")

  # Better.
  add_custom_command(TARGET image_processing POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${GLEW_DLL}"
      "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/glew32.dll")
endif ()
