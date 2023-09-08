#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <fstream>
#include <utility>
#include <vector>


namespace vk {

  inline auto read_shader_file(const std::string& filename) -> std::vector<char>
  {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file)
    {
      std::cerr << "  [VK] Failed to open file!\n";
      return {};
    }

    const auto file_size = static_cast<std::size_t>(file.tellg());
    auto buffer = std::vector<char>(file_size);

    file.seekg(0);
    file.read(buffer.data(), file_size);
    file.close();

    return buffer;
  }

  inline auto create_shader_module(const std::vector<char>& buffer)
      -> std::pair<VkShaderModule, bool>
  {
    auto create_info = VkShaderModuleCreateInfo{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = static_cast<std::uint32_t>(buffer.size());
    create_info.pCode = reinterpret_cast<const std::uint32_t*>(buffer.data());

    auto shader_module = VkShaderModule{};
    if (vkCreateShaderModule(_device, &create_info, nullptr, &shader_module) !=
        VK_SUCCESS)
    {
      return {shader_module, false};
    }

    return {shader_module, true};
  }


}  // namespace vk
