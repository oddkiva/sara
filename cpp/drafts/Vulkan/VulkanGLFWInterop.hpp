#pragma once

#include <vector>


namespace DO::Kalpana {

  //! Pre-condition: call glfwInit() first.
  auto list_required_vulkan_extensions_from_glfw() -> std::vector<const char*>;

}  // namespace DO::Kalpana
