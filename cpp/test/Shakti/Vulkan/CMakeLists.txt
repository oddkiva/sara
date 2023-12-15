file(GLOB test_Vulkan_SOURCE_FILES FILES test_*.cpp)

foreach(file ${test_Vulkan_SOURCE_FILES})
  shakti_add_test(
    SOURCES
    ${file}
    DEPENDENCIES
    DO::Sara::Core
    DO::Shakti::Vulkan
    FOLDER
    Vulkan)
endforeach()

# Compile the shader code first.
add_custom_command(
  TARGET shakti_test_vulkan_shader
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          $<TARGET_FILE_DIR:shakti_test_vulkan_shader>/test_shaders
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert -o
          $<TARGET_FILE_DIR:shakti_test_vulkan_shader>/test_shaders/vert.spv
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag -o
          $<TARGET_FILE_DIR:shakti_test_vulkan_shader>/test_shaders/frag.spv)

add_dependencies(shakti_test_vulkan_graphics_pipeline shakti_test_vulkan_shader)
add_dependencies(shakti_test_vulkan_graphics_backend shakti_test_vulkan_shader)
