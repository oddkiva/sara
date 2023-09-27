file(GLOB test_Vulkan_SOURCE_FILES FILES test_*.cpp)

foreach(file ${test_Vulkan_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME
    ${filename}
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
  TARGET test_vulkan_shader 
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_vulkan_shader>/test_shaders
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert -o
          $<TARGET_FILE_DIR:test_vulkan_shader>/test_shaders/vert.spv
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag -o
          $<TARGET_FILE_DIR:test_vulkan_shader>/test_shaders/frag.spv)

add_dependencies(test_vulkan_graphics_pipeline test_vulkan_shader) 
add_dependencies(test_vulkan_graphics_backend test_vulkan_shader)
