add_executable(hello_vulkan hello_vulkan.cpp)
target_link_libraries(
  hello_vulkan #
  PRIVATE DO::Sara::Core #
          DO::Shakti::Vulkan)
set_target_properties(hello_vulkan PROPERTIES FOLDER "Examples/Shakti/Vulkan")

add_custom_target(compile_hello_vulkan_shaders)
set_target_properties(compile_hello_vulkan_shaders PROPERTIES FOLDER "Examples/Shakti/Vulkan")
add_custom_command(
  TARGET compile_hello_vulkan_shaders
  COMMAND ${CMAKE_COMMAND} -E make_directory
          $<TARGET_FILE_DIR:hello_vulkan>/hello_vulkan_shaders
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert -o
          $<TARGET_FILE_DIR:hello_vulkan>/hello_vulkan_shaders/vert.spv
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag -o
          $<TARGET_FILE_DIR:hello_vulkan>/hello_vulkan_shaders/frag.spv)

# file(GLOB SHADER_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.vert *.frag)
add_custom_command(
  TARGET hello_vulkan
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          $<TARGET_FILE_DIR:hello_vulkan>/hello_vulkan_shaders
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert -o
          $<TARGET_FILE_DIR:hello_vulkan>/hello_vulkan_shaders/vert.spv
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag -o
          $<TARGET_FILE_DIR:hello_vulkan>/hello_vulkan_shaders/frag.spv)
