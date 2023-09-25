#pragma once


#include <vulkan/vulkan_core.h>

#include <fmt/format.h>


namespace DO::Shakti::Vulkan {

  class Buffer
  {
  public:
    Buffer() = default;

    Buffer(const VkDevice device, const VkDeviceSize size,
           const VkBufferUsageFlags usage)
      : _device{device}
    {
      auto create_info = VkBufferCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      create_info.size = size;
      create_info.usage = usage;
      create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;  // for now.

      const auto status =
          vkCreateBuffer(_device, &create_info, nullptr, &_handle);
      if (status != VK_SUCCESS)
      {
        throw std::runtime_error{
            fmt::format("[VK] Error: failed to create buffer! Error code: {}",
                        static_cast<int>(status))};
      }
    }

    Buffer(const Buffer&) = delete;

    Buffer(Buffer& other)
    {
      swap(other);
    }

    ~Buffer()
    {
      vkDestroyBuffer(_device, _handle, nullptr);
    }

    auto operator=(const Buffer&) -> Buffer& = delete;

    auto operator=(Buffer&& other) -> Buffer&
    {
      swap(other);
      return *this;
    }

    auto swap(Buffer& other) -> void
    {
      std::swap(_device, other._device);
      std::swap(_handle, other._handle);
    }

    auto get_memory_requirements() const -> VkMemoryRequirements
    {
      auto mem_requirements = VkMemoryRequirements{};
      vkGetBufferMemoryRequirements(_device, _handle, &mem_requirements);
      return mem_requirements;
    }

    auto bind(VkDeviceMemory device_memory, const std::uint32_t offset) -> void
    {
      const auto status =
          vkBindBufferMemory(_device, _handle, device_memory, offset);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Failed to bind buffer to device memory region! "
                        "Error code: {}",
                        static_cast<int>(status))};
    }

    operator VkBuffer&()
    {
      return _handle;
    }

    operator VkBuffer() const
    {
      return _handle;
    }

  private:
    VkDevice _device = nullptr;
    VkBuffer _handle = nullptr;
  };

}  // namespace DO::Shakti::Vulkan
