add_executable(hello_vulkan main.cpp)
target_include_directories(hello_vulkan
                           PRIVATE ${CMAKE_SOURCE_DIR}/cpp/third-party/eigen)
target_link_libraries(
  hello_vulkan
  PRIVATE glfw #
          DO_Sara_Core #
          fmt::fmt #
          Boost::filesystem #
          Boost::system #
          Vulkan::Vulkan)
set_target_properties(hello_vulkan PROPERTIES FOLDER "Tutorials")

# file(GLOB SHADER_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.vert *.frag)
add_custom_command(
  TARGET hello_vulkan
  PRE_BUILD
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert -o
          $<TARGET_FILE_DIR:hello_vulkan>/vert.spv
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag -o
          $<TARGET_FILE_DIR:hello_vulkan>/frag.spv)

