#pragma once

#include <vulkan/vulkan_core.h>

#include <fmt/format.h>

#include <algorithm>


namespace DO::Shakti::Vulkan {

  struct Semaphore
  {
    Semaphore(VkDevice device)
      : _device{device}
    {
      auto create_info = VkSemaphoreCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      const auto status =
          vkCreateSemaphore(device, &create_info, nullptr, &_handle);

      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Failed to create semaphore! Error code: {}",
                        static_cast<int>(status))};
    }

    Semaphore(const Semaphore&) = delete;

    Semaphore(Semaphore&& other)
    {
      std::swap(_device, other._device);
      std::swap(_handle, other._handle);
    }

    ~Semaphore()
    {
      vkDestroySemaphore(_device, _handle, nullptr);
    }

    auto operator=(const Semaphore&) -> Semaphore& = delete;

    VkDevice _device = nullptr;
    VkSemaphore _handle = nullptr;
  };

}  // namespace DO::Shakti::Vulkan
