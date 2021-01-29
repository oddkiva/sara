find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

find_program(GLSLC_EXECUTABLE NAMES glslc)

file(GLOB SHADER_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.vert *.frag)
add_custom_target(hello_vulkan_compile_shaders
  SOURCES ${SHADER_FILES})

add_custom_command(
  TARGET hello_vulkan_compile_shaders
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert -o
  ${CMAKE_BINARY_DIR}/vert.spv
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag -o
  ${CMAKE_BINARY_DIR}/frag.spv)


add_executable(hello_vulkan main.cpp)
target_link_libraries(hello_vulkan PRIVATE
  glfw
  Vulkan::Vulkan)
target_compile_features(hello_vulkan PRIVATE cxx_std_17)


add_dependencies(hello_vulkan hello_vulkan_compile_shaders)
