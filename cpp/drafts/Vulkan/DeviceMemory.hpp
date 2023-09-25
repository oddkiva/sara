#pragma once

#include <vulkan/vulkan_core.h>

#include <fmt/format.h>

#include <stdexcept>


namespace DO::Shakti::Vulkan {

  class DeviceMemory
  {
  public:
    DeviceMemory() = default;

    DeviceMemory(VkDevice device, const std::uint32_t size,
                 const std::uint32_t memory_type_index)
      : _device{device}
    {
      auto allocate_info = VkMemoryAllocateInfo{};
      allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocate_info.allocationSize = size;
      allocate_info.memoryTypeIndex = memory_type_index;
      const auto status =
          vkAllocateMemory(_device, &allocate_info, nullptr, &_handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Failed to allocate device memory! Error code: {}",
                        static_cast<int>(status))};
    }

    ~DeviceMemory()
    {
      if (_device == nullptr || _handle == nullptr)
        return;

      vkFreeMemory(_device, _handle, nullptr);
    }

    operator VkDeviceMemory&()
    {
      return _handle;
    }

    operator VkDeviceMemory() const
    {
      return _handle;
    }

  private:
    VkDevice _device = nullptr;
    VkDeviceMemory _handle = nullptr;
  };

}  // namespace DO::Shakti::Vulkan
