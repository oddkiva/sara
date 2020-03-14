file(GLOB SRC_FILES FILES *.cpp *.hpp *.cl)
source_group("OpenCL Source Files" REGULAR_EXPRESSION
  "${CMAKE_CURRENT_SOURCE_DIR}/.*\\.cl")

add_executable(image_processing ${SRC_FILES})
target_include_directories(image_processing PRIVATE
  ${OpenCL_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR})
target_compile_definitions(image_processing PRIVATE
  -DCL_SILENCE_DEPRECATION
  -DGL_SILENCE_DEPRECATION)
target_link_libraries(image_processing PRIVATE
  ${DO_Sara_LIBRARIES}
  ${OpenCL_LIBRARY}
  ${OPENGL_LIBRARIES}
  glfw)
if (NOT APPLE)
  target_link_libraries(image_processing PRIVATE ${GLEW_LIBRARIES})
endif ()
set_target_properties(image_processing PROPERTIES
  COMPILE_FLAGS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}")
set_property(TARGET image_processing PROPERTY FOLDER "Tutorials/OpenGL")


if (MSVC)
  # Works but cumbersome.
  #install(FILES "${DIRS}/glew32.dll" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/\${CMAKE_INSTALL_CONFIG_NAME}")

  # Better.
  add_custom_command(TARGET image_processing POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${GLEW_DLL}"
      "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/glew32.dll")
endif ()
