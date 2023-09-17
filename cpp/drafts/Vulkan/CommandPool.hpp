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


namespace DO::Shakti::Vulkan {

  struct CommandPool
  {
    CommandPool(const Device& device, const std::uint32_t queue_index)
      : device_handle{device.handle}
    {
      auto create_info = VkCommandPoolCreateInfo{};
      {
        create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        create_info.queueFamilyIndex = queue_index;
        create_info.flags = 0;
      }

      const auto status =
          vkCreateCommandPool(device_handle, &create_info, nullptr, &handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("Error: failed to create command pool! Error code: {}",
                        static_cast<int>(status))};
    }

    CommandPool(const CommandPool& other) = delete;

    CommandPool(CommandPool&& other)
    {
      std::swap(device_handle, other.device_handle);
      std::swap(handle, other.handle);
    }

    ~CommandPool()
    {
      if (handle != nullptr)
      {
        SARA_DEBUG << fmt::format("[VK] Destroying command pool: {}...\n",
                                  fmt::ptr(handle));
        vkDestroyCommandPool(device_handle, handle, nullptr);
      }
    }

    auto operator=(const CommandPool& other) -> CommandPool& = delete;

    VkDevice device_handle = nullptr;
    VkCommandPool handle = nullptr;
  };

}  // namespace DO::Shakti::Vulkan
