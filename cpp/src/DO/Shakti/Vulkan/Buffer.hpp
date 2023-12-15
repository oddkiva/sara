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
#include <DO/Shakti/Vulkan/CommandPool.hpp>

#include <fmt/format.h>

#include <cstdint>
#include <limits>
#include <vulkan/vulkan_core.h>


namespace DO::Shakti::Vulkan {

  class Buffer
  {
  public:
    Buffer() = default;

    Buffer(const VkDevice device, const VkDeviceSize size,
           const VkBufferUsageFlags usage)
      : _device{device}
      , _size{size}
    {
      auto create_info = VkBufferCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      create_info.size = size;
      create_info.usage = usage;
      create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;  // for now.

      const auto status =
          vkCreateBuffer(_device, &create_info, nullptr, &_handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Error: failed to create buffer! Error code: {}",
                        static_cast<int>(status))};
    }

    Buffer(const Buffer&) = delete;

    Buffer(Buffer& other)
    {
      swap(other);
    }

    ~Buffer()
    {
      if (_device == nullptr || _handle == nullptr)
        return;
      vkDestroyBuffer(_device, _handle, nullptr);
    }

    auto operator=(const Buffer&) -> Buffer& = delete;

    auto operator=(Buffer&& other) -> Buffer&
    {
      swap(other);
      return *this;
    }

    operator VkBuffer&()
    {
      return _handle;
    }

    operator VkBuffer() const
    {
      return _handle;
    }

    auto size() const -> VkDeviceSize
    {
      return _size;
    }

    auto swap(Buffer& other) -> void
    {
      std::swap(_device, other._device);
      std::swap(_handle, other._handle);
      std::swap(_size, other._size);
    }

    auto memory_requirements() const -> VkMemoryRequirements
    {
      auto mem_requirements = VkMemoryRequirements{};
      vkGetBufferMemoryRequirements(_device, _handle, &mem_requirements);
      return mem_requirements;
    }

    auto bind(VkDeviceMemory device_memory, const std::uint32_t offset) const
        -> void
    {
      const auto status =
          vkBindBufferMemory(_device, _handle, device_memory, offset);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Failed to bind buffer to device memory region! "
                        "Error code: {}",
                        static_cast<int>(status))};
    }

  private:
    VkDevice _device = nullptr;
    VkBuffer _handle = nullptr;
    VkDeviceSize _size = 0;
  };


  struct BufferFactory
  {
    template <typename T>
    inline auto make_staging_buffer(const std::size_t n) const -> Buffer
    {
      const auto byte_size = sizeof(T) * n;
      return Buffer(device, byte_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    }

    template <typename T>
    inline auto make_uniform_buffer(const std::size_t n) const -> Buffer
    {
      const auto byte_size = sizeof(T) * n;
      return Buffer(device, byte_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    }

    template <typename T>
    inline auto make_device_vertex_buffer(const std::size_t n) const -> Buffer
    {
      const auto byte_size = sizeof(T) * n;
      return Buffer(device, byte_size,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    }

    template <typename T>
    inline auto make_device_index_buffer(const std::size_t n) const -> Buffer
    {
      const auto byte_size = sizeof(T) * n;
      return Buffer(device, byte_size,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    }

    const VkDevice device = nullptr;
  };


  inline auto record_copy_buffer(const Buffer& src, const Buffer& dst,
                                 const VkCommandBuffer cmd_buffer) -> void
  {
    // Specify the copy operation for this command buffer.
    auto cmd_buf_begin_info = VkCommandBufferBeginInfo{};
    cmd_buf_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmd_buf_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_buffer, &cmd_buf_begin_info);
    {
      auto region = VkBufferCopy{};
      region.size = src.size();
      vkCmdCopyBuffer(cmd_buffer, src, dst, 1, &region);
    }
    vkEndCommandBuffer(cmd_buffer);
  }

}  // namespace DO::Shakti::Vulkan
