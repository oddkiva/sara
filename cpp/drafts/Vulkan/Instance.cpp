#include <drafts/Vulkan/Instance.hpp>


namespace svk = DO::Shakti::EasyVulkan;

auto svk::list_available_instance_layers() -> std::vector<VkLayerProperties>
{
  auto layer_count = std::uint32_t{};
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
  if (layer_count == 0)
    return {};

  auto available_layers = std::vector<VkLayerProperties>(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

  return available_layers;
}
