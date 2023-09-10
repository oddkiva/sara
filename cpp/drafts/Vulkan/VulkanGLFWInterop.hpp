#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#include <vector>


namespace DO::Kalpana {

  inline auto list_required_vulkan_extensions_from_glfw(
      const bool add_debug_utility_extension) -> std::vector<const char*>
  {
    auto glfw_extension_count = std::uint32_t{};
    const char** glfw_extensions = nullptr;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    std::vector<const char*> extensions(glfw_extensions,
                                        glfw_extensions + glfw_extension_count);

    if (add_debug_utility_extension)
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return extensions;
  }

}  // namespace DO::Kalpana
