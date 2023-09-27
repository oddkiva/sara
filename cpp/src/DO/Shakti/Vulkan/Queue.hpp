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

#include <DO/Shakti/Vulkan/CommandBuffer.hpp>
#include <DO/Shakti/Vulkan/Device.hpp>

#include <fmt/core.h>


namespace DO::Shakti::Vulkan {

  class Queue
  {
  public:
    Queue() = default;

    Queue(const Device& device, const std::uint32_t queue_index)
    {
      vkGetDeviceQueue(device, queue_index, 0, &_handle);
    }

    auto submit(const VkSubmitInfo& submit_info, const VkFence fence) const
        -> void
    {
      const auto status = vkQueueSubmit(_handle, 1, &submit_info, fence);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Error: failed to submit command buffer sequence "
                        "to the queue! Error code: {}",
                        static_cast<int>(status))};
    };

    auto submit_copy_commands(const CommandBufferSequence& copy_cmd_bufs) const
        -> void
    {
      auto submit_info = VkSubmitInfo{};
      submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit_info.commandBufferCount =
          static_cast<std::uint32_t>(copy_cmd_bufs.size());
      submit_info.pCommandBuffers = copy_cmd_bufs.data();
      submit(submit_info, VK_NULL_HANDLE);
    }

    auto wait() const -> void
    {
      vkQueueWaitIdle(_handle);
    }

    operator VkQueue&()
    {
      return _handle;
    }

    operator VkQueue() const
    {
      return _handle;
    }

  private:
    VkQueue _handle = nullptr;
  };

}  // namespace DO::Shakti::Vulkan
