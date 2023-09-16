// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <drafts/Vulkan/Device.hpp>

#include <cstdint>
#include <fstream>
#include <utility>
#include <vector>


namespace DO::Shakti::Vulkan {

  inline auto read_shader_file(const std::string& filename) -> std::vector<char>
  {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file)
      throw std::runtime_error{fmt::format(
          "Error: failed to open compiled shader file: {}", filename)};

    const auto file_size = static_cast<std::size_t>(file.tellg());
    auto buffer = std::vector<char>(file_size);

    file.seekg(0);
    file.read(buffer.data(), file_size);
    file.close();

    return buffer;
  }

}  // namespace DO::Shakti::Vulkan


namespace DO::Shakti::Vulkan {

  struct ShaderModule
  {
    ShaderModule() = default;

    ShaderModule(const VkDevice device, const std::vector<char>& shader_source)
      : device_handle{device}
    {
      auto create_info = VkShaderModuleCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      create_info.codeSize = static_cast<std::uint32_t>(shader_source.size());
      create_info.pCode =
          reinterpret_cast<const std::uint32_t*>(shader_source.data());

      const auto status =
          vkCreateShaderModule(device_handle, &create_info, nullptr, &handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("Failed to create shader module! Error code: {}",
                        static_cast<int>(status))};
    }

    ShaderModule(const ShaderModule&) = delete;

    ShaderModule(ShaderModule&& other) = default;

    ~ShaderModule()
    {
      if (handle == nullptr)
        return;
      vkDestroyShaderModule(device_handle, handle, nullptr);
    }

    auto operator=(const ShaderModule&) -> ShaderModule& = delete;

    auto operator=(ShaderModule&& other) -> ShaderModule& = default;

  public:
    VkDevice device_handle = nullptr;
    VkShaderModule handle = nullptr;
  };

}  // namespace DO::Shakti::Vulkan
