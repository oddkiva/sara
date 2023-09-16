file(GLOB test_EasyVulkan_SOURCE_FILES FILES test_*.cpp)

foreach(file ${test_EasyVulkan_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME
    ${filename}
    SOURCES
    ${file}
    DEPENDENCIES
    EasyVulkan
    glfw
    DO::Sara::Core
    FOLDER
    Vulkan)
endforeach()

# Compile the shader code first.
add_custom_target(compile_test_vulkan_shaders)
add_custom_command(
  TARGET compile_test_vulkan_shaders
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert -o
          ${CMAKE_BINARY_DIR}/vert.spv
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag -o
          ${CMAKE_BINARY_DIR}/frag.spv
  COMMENT "Compiling test Vulkan/GLSL shaders")

add_dependencies(test_vulkan_graphics_pipeline compile_test_vulkan_shaders)
