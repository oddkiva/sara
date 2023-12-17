add_executable(hello_vulkan_image main.cpp)
target_link_libraries(
  hello_vulkan_image
  PRIVATE SignalHandler #
          DO::Sara::Core #
          DO::Sara::ImageProcessing #
          DO::Sara::VideoIO #
          DO::Shakti::Vulkan)
set_target_properties(hello_vulkan_image PROPERTIES FOLDER
                                                    "Examples/Shakti/Vulkan")

add_custom_target(compile_hello_vulkan_image_shaders)
set_target_properties(compile_hello_vulkan_image_shaders
                      PROPERTIES FOLDER "Examples/Shakti/Vulkan")
add_custom_command(
  TARGET compile_hello_vulkan_image_shaders
  COMMAND ${CMAKE_COMMAND} -E make_directory
          $<TARGET_FILE_DIR:hello_vulkan_image>/hello_vulkan_image_shaders
  COMMAND
    ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert -o
    $<TARGET_FILE_DIR:hello_vulkan_image>/hello_vulkan_image_shaders/vert.spv
  COMMAND
    ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag -o
    $<TARGET_FILE_DIR:hello_vulkan_image>/hello_vulkan_image_shaders/frag.spv)

add_custom_command(
  TARGET hello_vulkan_image
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          $<TARGET_FILE_DIR:hello_vulkan_image>/hello_vulkan_image_shaders
  COMMAND
    ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert -o
    $<TARGET_FILE_DIR:hello_vulkan_image>/hello_vulkan_image_shaders/vert.spv
  COMMAND
    ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag -o
    $<TARGET_FILE_DIR:hello_vulkan_image>/hello_vulkan_image_shaders/frag.spv)
