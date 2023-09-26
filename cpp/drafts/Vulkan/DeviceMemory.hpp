#pragma once

#include <drafts/Vulkan/Buffer.hpp>
#include <drafts/Vulkan/PhysicalDevice.hpp>

#include <fmt/format.h>

#include <stdexcept>
#include <vulkan/vulkan_core.h>


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
    auto copy_from(T* src_ptr, const VkDeviceSize src_num_elements,
                   const VkDeviceSize dst_start = 0) const -> void
    {
      const auto src_byte_size = sizeof(T) * src_num_elements;

      // Get the virtual host destination pointer.
      auto dst_ptr = static_cast<void*>(nullptr);
      vkMapMemory(_device, _handle, sizeof(T) * dst_start, src_byte_size,
                  0 /* flags */, &dst_ptr);
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

  struct DeviceMemoryFactory
  {
    auto allocate_for_staging_buffer(const Buffer& buffer) const -> DeviceMemory
    {
      static constexpr auto mem_props = VkMemoryPropertyFlags{
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT  //
      };

      const auto mem_reqs = buffer.get_memory_requirements();
      const auto mem_type =
          _physical_device.find_memory_type(mem_reqs.memoryTypeBits, mem_props);

      return {_device, buffer.size(), mem_type};
    }

    auto allocate_for_device_buffer(const Buffer& buffer) const -> DeviceMemory
    {
      static constexpr auto mem_props =
          VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};

      const auto mem_reqs = buffer.get_memory_requirements();
      const auto mem_type =
          _physical_device.find_memory_type(mem_reqs.memoryTypeBits, mem_props);

      SARA_CHECK(buffer.size());

      return {_device, buffer.size(), mem_type};
    }

    const PhysicalDevice& _physical_device = nullptr;
    const VkDevice _device = nullptr;
  };

}  // namespace DO::Shakti::Vulkan
