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

#include <vulkan/vulkan_core.h>

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

    Image(Image&& other);

    ~Image()
    {
      if (_device == nullptr || _handle == nullptr)
        return;
      vkDestroyImage(_device, _handle, nullptr);
    }

    auto operator=(const Image&) -> Image& = delete;

    auto operator=(Image&& other) -> Image&;

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

  private:
    VkDevice _device = VK_NULL_HANDLE;
    VkImage _handle = VK_NULL_HANDLE;
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

    auto sizes(const std::array<std::uint32_t, 2>& value) -> Builder&
    {
      _sizes[0] = value[0];
      _sizes[1] = value[1];
      return *this;
    }

    auto sizes(const std::array<std::uint32_t, 3>& value) -> Builder&
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

    auto create() const -> Image
    {
      auto create_info = VkImageCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      create_info.imageType = _image_type;
      create_info.extent.width = _sizes[0];
      create_info.extent.height = _sizes[1];
      create_info.extent.depth = _sizes[2];
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
    //! @brief Image sizes: [width, height, depth]
    std::array<std::uint32_t, 3> _sizes = {0, 0, 1};
    VkImageType _image_type = VK_IMAGE_TYPE_2D;
    VkFormat _format;
    VkImageTiling _tiling;
    std::uint32_t _mip_levels = 1;
    std::uint32_t _array_layers = 1;
    VkImageLayout _initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::uint32_t _usage;
    VkSampleCountFlagBits _samples = VK_SAMPLE_COUNT_1_BIT;
    VkSharingMode _sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
    VkImageUsageFlags _flags;
  };


}  // namespace DO::Shakti::Vulkan
