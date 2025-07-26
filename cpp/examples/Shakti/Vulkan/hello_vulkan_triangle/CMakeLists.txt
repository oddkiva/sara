add_executable(hello_vulkan_triangle main.cpp)
target_link_libraries(
  hello_vulkan_triangle #
  PRIVATE DO::Sara::Core #
          DO::Shakti::Vulkan #
          SignalHandler)
set_target_properties(hello_vulkan_triangle PROPERTIES FOLDER
                                                       "Examples/Shakti/Vulkan")

add_custom_target(compile_hello_vulkan_triangle_shaders)
add_custom_command(
  TARGET compile_hello_vulkan_triangle_shaders
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          $<TARGET_FILE_DIR:hello_vulkan_triangle>/hello_vulkan_triangle_shaders
  COMMAND
    ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert -o
    $<TARGET_FILE_DIR:hello_vulkan_triangle>/hello_vulkan_triangle_shaders/vert.spv
  COMMAND
    ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag -o
    $<TARGET_FILE_DIR:hello_vulkan_triangle>/hello_vulkan_triangle_shaders/frag.spv
)
set_target_properties(compile_hello_vulkan_triangle_shaders
                      PROPERTIES FOLDER "Examples/Shakti/Vulkan")

# file(GLOB SHADER_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.vert *.frag)
add_custom_command(
  TARGET hello_vulkan_triangle
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          $<TARGET_FILE_DIR:hello_vulkan_triangle>/hello_vulkan_triangle_shaders
  COMMAND
    ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert -o
    $<TARGET_FILE_DIR:hello_vulkan_triangle>/hello_vulkan_triangle_shaders/vert.spv
  COMMAND
    ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag -o
    $<TARGET_FILE_DIR:hello_vulkan_triangle>/hello_vulkan_triangle_shaders/frag.spv
)
