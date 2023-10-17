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

#include <DO/Shakti/Vulkan/Buffer.hpp>

#include <fmt/core.h>

#include <stdexcept>


namespace DO::Shakti::Vulkan {

  class Image
  {
  public:
    class Builder;
    friend class Builder;

    Image() = default;

    Image(const Image&) = delete;

    Image(Image&&) = default;

    ~Image()
    {
      if (_device == nullptr || _handle == nullptr)
        return;
      vkDestroyImage(_device, _handle, nullptr);
    }

    auto operator=(const Image&) -> Image& = delete;

    auto operator=(Image&& other) -> Image&
    {
      _device = std::move(other._device);
      _handle = std::move(other._handle);
      _sizes = std::move(other._sizes);
      return *this;
    }

    operator VkImage&()
    {
      return _handle;
    }

    operator VkImage() const
    {
      return _handle;
    }

    auto memory_requirements() const -> VkMemoryRequirements
    {
      auto mem_reqs = VkMemoryRequirements{};
      vkGetImageMemoryRequirements(_device, _handle, &mem_reqs);
      return mem_reqs;
    }

    auto bind(VkDeviceMemory device_memory, const std::uint32_t offset) const
        -> void
    {
      const auto status =
          vkBindImageMemory(_device, _handle, device_memory, offset);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Failed to bind image to device memory region! "
                        "Error code: {}",
                        static_cast<int>(status))};
    }

    auto sizes() const -> const VkExtent3D&
    {
      return _sizes;
    }

  private:
    VkDevice _device = VK_NULL_HANDLE;
    VkImage _handle = VK_NULL_HANDLE;
    VkExtent3D _sizes = {.width = 0, .height = 0, .depth = 0};
  };


  class Image::Builder
  {
  public:
    Builder() = default;

    Builder(VkDevice device)
      : _device{device}
    {
    }

    auto image_type(const VkImageType value) -> Builder&
    {
      _image_type = value;
      return *this;
    }

    auto sizes(const VkExtent2D& value) -> Builder&
    {
      _sizes.width = value.width;
      _sizes.height = value.height;
      _sizes.depth = 1;
      return *this;
    }

    auto sizes(const VkExtent3D& value) -> Builder&
    {
      _sizes = value;
      return *this;
    }

    auto format(const VkFormat value) -> Builder&
    {
      _format = value;
      return *this;
    }

    auto tiling(const VkImageTiling value) -> Builder&
    {
      _tiling = value;
      return *this;
    }

    auto mip_levels(const std::uint32_t value) -> Builder&
    {
      _mip_levels = value;
      return *this;
    }

    auto initial_layout(const VkImageLayout value) -> Builder&
    {
      _initial_layout = value;
      return *this;
    }

    auto usage(const VkImageUsageFlags value) -> Builder&
    {
      _usage = value;
      return *this;
    }

    auto create() const -> Image
    {
      auto create_info = VkImageCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      create_info.imageType = _image_type;
      create_info.extent = _sizes;
      create_info.mipLevels = _mip_levels;
      create_info.arrayLayers = _array_layers;
      create_info.format = _format;
      create_info.tiling = _tiling;
      create_info.initialLayout = _initial_layout;
      create_info.usage = _usage;
      create_info.samples = _samples;
      create_info.sharingMode = _sharing_mode;

      auto image = Image{};
      image._device = _device;
      image._sizes = _sizes;
      const auto status = vkCreateImage(_device, &create_info, nullptr,  //
                                        &image._handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Failed to create image! Error code: {}",
                        static_cast<int>(status))};

      return image;
    }

  private:
    VkDevice _device = VK_NULL_HANDLE;
    VkExtent3D _sizes = {.width = 0, .height = 0, .depth = 1};
    VkImageType _image_type = VK_IMAGE_TYPE_2D;
    VkFormat _format;       // VK_FORMAT_R8G8B8A8_SRGB
    VkImageTiling _tiling;  // VK_IMAGE_TILING_OPTIMAL
    std::uint32_t _mip_levels = 1;
    std::uint32_t _array_layers = 1;
    VkImageLayout _initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageUsageFlags _usage;  // VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               // VK_IMAGE_USAGE_SAMPLED_BIT
    VkSampleCountFlagBits _samples = VK_SAMPLE_COUNT_1_BIT;
    VkSharingMode _sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
  };


  inline auto record_copy_buffer_to_image(const Buffer& src, const Image& dst,
                                          const VkCommandBuffer cmd_buffer)
      -> void
  {
    // Specify the copy operation for this command buffer.
    auto cmd_buf_begin_info = VkCommandBufferBeginInfo{};
    cmd_buf_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmd_buf_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmd_buffer, &cmd_buf_begin_info);
    {
      auto region = VkBufferImageCopy{};
      // Source region.
      region.bufferOffset = 0;
      region.bufferRowLength = 0;
      region.bufferImageHeight = 0;

      // Destination region.
      region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.imageSubresource.mipLevel = 0;
      region.imageSubresource.baseArrayLayer = 0;
      region.imageSubresource.layerCount = 1;
      region.imageOffset = {0, 0, 0};
      region.imageExtent = dst.sizes();

      vkCmdCopyBufferToImage(cmd_buffer, src, dst,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }
    vkEndCommandBuffer(cmd_buffer);
  }

  inline auto record_image_layout_transition(const VkImage image,
                                             const VkImageLayout old_layout,
                                             const VkImageLayout new_layout,
                                             const VkCommandBuffer cmd_buffer)
      -> void
  {
    // Specify the copy operation for this command buffer.
    auto cmd_buf_begin_info = VkCommandBufferBeginInfo{};
    cmd_buf_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmd_buf_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmd_buffer, &cmd_buf_begin_info);
    {
      auto barrier = VkImageMemoryBarrier{};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      //
      barrier.oldLayout = old_layout;
      barrier.newLayout = new_layout;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image;
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barrier.subresourceRange.baseMipLevel = 0;
      barrier.subresourceRange.levelCount = 1;
      barrier.subresourceRange.baseArrayLayer = 0;
      barrier.subresourceRange.layerCount = 1;

      auto src_stage = VkPipelineStageFlags{};
      auto dst_stage = VkPipelineStageFlags{};

      if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
          new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
      {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      }
      else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
      {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      }
      else
        throw std::invalid_argument{
            "[VK] Error: unimplemented/unsupported layout transition!"  //
        };

      vkCmdPipelineBarrier(cmd_buffer, src_stage, dst_stage, 0, 0, nullptr, 0,
                           nullptr, 1, &barrier);
    }
    vkEndCommandBuffer(cmd_buffer);
  }


}  // namespace DO::Shakti::Vulkan
