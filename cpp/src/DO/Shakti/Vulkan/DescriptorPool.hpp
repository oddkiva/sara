#pragma once

#include <vulkan/vulkan_core.h>

#include <fmt/core.h>


namespace DO::Shakti::Vulkan {

  class DescriptorPool;
  class DescriptorSet;

  class DescriptorPool
  {
    friend class DescriptorSet;

  public:
    DescriptorPool() = default;

    DescriptorPool(const DescriptorPool&) = delete;

    DescriptorPool(DescriptorPool&& other)
    {
      swap(other);
    }

    DescriptorPool(VkDevice device)
      : _device{device}
    {
      auto create_info = VkDescriptorPoolCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      const auto status =
          vkCreateDescriptorPool(device, &create_info, nullptr, &_handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{fmt::format(
            "[VK] Error: failed to create descriptor pool! Error code: {}",
            static_cast<int>(status))};
    }

    ~DescriptorPool()
    {
      if (_device == nullptr || _handle == nullptr)
        return;
      vkDestroyDescriptorPool(_device, _handle, nullptr);
    }

    auto operator=(const DescriptorPool&) -> DescriptorPool& = delete;

    auto operator=(DescriptorPool&& other) -> DescriptorPool&
    {
      swap(other);
      return *this;
    }

    operator VkDescriptorPool&()
    {
      return _handle;
    }

    operator VkDescriptorPool() const
    {
      return _handle;
    }

    auto swap(DescriptorPool& other) -> void
    {
      std::swap(_device, other._device);
      std::swap(_handle, other._handle);
    }

  private:
    VkDevice _device = nullptr;
    VkDescriptorPool _handle = nullptr;
  };

}  // namespace DO::Shakti::Vulkan
