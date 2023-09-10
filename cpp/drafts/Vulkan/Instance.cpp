#include <drafts/Vulkan/Instance.hpp>


namespace DO::Kalpana::EasyVulkan {

  auto list_available_instance_layers() -> std::vector<VkLayerProperties>
  {
    auto layer_count = std::uint32_t{};
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    if (layer_count == 0)
      return {};

    auto available_layers = std::vector<VkLayerProperties>(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    return available_layers;
  }

  auto check_validation_layer_support(
      const std::vector<const char*>& requested_validation_layers) -> bool
  {
    // First edge case.
    if (requested_validation_layers.empty())
      return true;

    const auto available_layers = list_available_instance_layers();
    // Second edge case.
    if (available_layers.empty())
      return false;

    // General case.
    for (const auto layer_name : requested_validation_layers)
    {
      const auto available_layer_it = std::find_if(
          available_layers.begin(), available_layers.end(),
          [&layer_name](const auto& layer_properties) {
            return strcmp(layer_name, layer_properties.layerName) == 0;
          });
      const auto layer_found = available_layer_it != available_layers.end();

      std::cout << fmt::format("  [VK][Validation] {}: {}\n",  //
                               layer_name, layer_found ? "FOUND" : "NOT FOUND");

      if (!layer_found)
        return false;
    }

    return true;
  }

}  // namespace DO::Kalpana::EasyVulkan
