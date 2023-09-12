find_package(Vulkan REQUIRED)
find_program(
  GLSLC_EXECUTABLE
  NAMES glslc
  HINTS ENV VULKAN_SDK
  PATH_SUFFIXES bin)

add_library(
  EasyVulkan #
  Instance.cpp Instance.hpp #
  Surface.cpp Surface.hpp)
target_link_libraries(
  EasyVulkan
  PUBLIC Vulkan::Vulkan #
         glfw #
         DO::Sara::Core #
         fmt::fmt)

add_subdirectory(test)
