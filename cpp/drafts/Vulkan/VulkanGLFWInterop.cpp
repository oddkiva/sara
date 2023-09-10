#include <drafts/Vulkan/VulkanGLFWInterop.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <GLFW/glfw3.h>

#include <vulkan/vulkan_core.h>

#include <fmt/format.h>

#include <cstdint>


namespace k = DO::Kalpana;

auto k::list_required_vulkan_extensions_from_glfw() -> std::vector<const char*>
{
  auto glfw_extension_count = std::uint32_t{};
  const char** glfw_extensions = nullptr;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

  std::vector<const char*> extensions(glfw_extensions,
                                      glfw_extensions + glfw_extension_count);

  SARA_DEBUG << "Vulkan extensions required by GLFW:\n";
  for (const auto extension : extensions)
    SARA_DEBUG << fmt::format("- {}\n", extension);

  return extensions;
}
