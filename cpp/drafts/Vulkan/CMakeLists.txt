find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

find_program(
  GLSLC_EXECUTABLE
  NAMES glslc
  HINTS ENV VULKAN_SDK
  PATH_SUFFIXES bin)

add_executable(hello_vulkan main.cpp)
target_include_directories(hello_vulkan
                           PRIVATE ${CMAKE_SOURCE_DIR}/cpp/third-party/eigen)
target_link_libraries(
  hello_vulkan
  PRIVATE glfw #
          Boost::filesystem #
          Boost::system #
          Vulkan::Vulkan)
target_compile_features(hello_vulkan PRIVATE cxx_std_17)
set_target_properties(hello_vulkan PROPERTIES FOLDER "Tutorials")

# file(GLOB SHADER_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.vert *.frag)
add_custom_command(
  TARGET hello_vulkan
  PRE_BUILD
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.vert -o
          $<TARGET_FILE_DIR:hello_vulkan>/vert.spv
  COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shader.frag -o
          $<TARGET_FILE_DIR:hello_vulkan>/frag.spv)
