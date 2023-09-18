#pragma once

#include <vulkan/vulkan_core.h>

#include <fmt/format.h>

#include <algorithm>


namespace DO::Shakti::Vulkan {

  struct Fence
  {
    Fence(VkDevice device)
      : _device{device}
    {
      auto create_info = VkFenceCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

      const auto status = vkCreateFence(_device, &create_info, nullptr,  //
                                        &_handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Failed to create fence! Error code: {}",
                        static_cast<int>(status))};
    }

    Fence(const Fence&) = delete;

    Fence(Fence&& other)
    {
      std::swap(_device, other._device);
      std::swap(_handle, other._handle);
    }

    ~Fence()
    {
      vkDestroyFence(_device, _handle, nullptr);
    }

    auto operator=(const Fence&) -> Fence& = delete;

    auto wait(const std::uint64_t timeout_ns) -> void
    {
      vkWaitForFences(_device, 1, &_handle, VK_TRUE, timeout_ns);
    }

    VkDevice _device = nullptr;
    VkFence _handle = nullptr;
  };

}  // namespace DO::Shakti::Vulkan
