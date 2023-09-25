#pragma once

#include <vulkan/vulkan_core.h>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <fmt/format.h>

#include <stdexcept>


namespace DO::Shakti::Vulkan {

  class DeviceMemory
  {
  public:
    DeviceMemory() = default;

    DeviceMemory(const DeviceMemory&) = delete;

    DeviceMemory(DeviceMemory&& other)
    {
      swap(other);
    }

    DeviceMemory(VkDevice device, const VkDeviceSize size,
                 const std::uint32_t memory_type_index)
      : _device{device}
      , _size{size}
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

    auto operator=(const DeviceMemory&) -> DeviceMemory& = delete;

    auto operator=(DeviceMemory&& other) -> DeviceMemory&
    {
      swap(other);
      return *this;
    }

    auto size() const -> VkDeviceSize
    {
      return _size;
    }

    template <typename T>
    auto copy_from(T* src_ptr, const VkDeviceSize src_size,
                   const VkDeviceSize dst_offset = 0) -> void
    {
      const auto src_byte_size = sizeof(T) * src_size;

      // Get the virtual host destination pointer.
      auto dst_ptr = static_cast<void*>(nullptr);
      vkMapMemory(_device, _handle, dst_offset, src_byte_size, 0 /* flags */,
                  &dst_ptr);
      // Copy the host data to the virtual host destination.
      std::memcpy(dst_ptr, src_ptr, src_byte_size);
      // Invalidate the virtual host destination pointer.
      vkUnmapMemory(_device, _handle);
    }

    operator VkDeviceMemory&()
    {
      return _handle;
    }

    operator VkDeviceMemory() const
    {
      return _handle;
    }

    auto swap(DeviceMemory& other) -> void
    {
      std::swap(_device, other._device);
      std::swap(_handle, other._handle);
      std::swap(_size, other._size);
    }

  private:
    VkDevice _device = nullptr;
    VkDeviceMemory _handle = nullptr;
    VkDeviceSize _size = 0;
  };

}  // namespace DO::Shakti::Vulkan
