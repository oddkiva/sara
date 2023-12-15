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

  class ImageView
  {
  public:
    class Builder;
    friend class Builder;

    ImageView() = default;

    ImageView(const ImageView&) = delete;

    ImageView(ImageView&& other)
    {
      swap(other);
    }

    ~ImageView()
    {
      if (_device == nullptr || _handle == nullptr)
        return;
      vkDestroyImageView(_device, _handle, nullptr);
    }

    auto operator=(const ImageView&) -> ImageView& = delete;

    auto operator=(ImageView&& other) -> ImageView&
    {
      swap(other);
      return *this;
    }

    operator VkImageView&()
    {
      return _handle;
    }

    operator VkImageView() const
    {
      return _handle;
    }

    auto swap(ImageView& other) -> void
    {
      std::swap(_device, other._device);
      std::swap(_handle, other._handle);
    }

  private:
    VkDevice _device = VK_NULL_HANDLE;
    VkImageView _handle = VK_NULL_HANDLE;
  };


  class ImageView::Builder
  {
  public:
    explicit Builder(VkDevice device)
      : _device{device}
    {
    }

    auto image(const VkImage value) -> Builder&
    {
      _image = value;
      return *this;
    }

    auto view_type(const VkImageViewType value) -> Builder&
    {
      _view_type = value;
      return *this;
    }

    auto format(const VkFormat value) -> Builder&
    {
      _format = value;
      return *this;
    }

    auto aspect_mask(const VkImageAspectFlags value) -> Builder&
    {
      _aspect_mask = value;
      return *this;
    }

    auto create() const -> ImageView
    {
      auto create_info = VkImageViewCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      create_info.image = _image;
      create_info.viewType = _view_type;
      create_info.format = _format;
      create_info.subresourceRange.aspectMask = _aspect_mask;

      // TODO: Worry about this later.
      create_info.subresourceRange.baseMipLevel = 0;
      create_info.subresourceRange.levelCount = 1;
      create_info.subresourceRange.baseArrayLayer = 0;
      create_info.subresourceRange.layerCount = 1;

      auto image_view = ImageView{};
      image_view._device = _device;

      const auto status = vkCreateImageView(_device, &create_info, nullptr,
                                            &image_view._handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Failed to create image view! Error code: {}",
                        static_cast<int>(status))};

      return image_view;
    }

  private:
    VkDevice _device = VK_NULL_HANDLE;
    VkImage _image = VK_NULL_HANDLE;
    VkImageViewType _view_type = VK_IMAGE_VIEW_TYPE_2D;
    VkFormat _format;
    VkImageAspectFlags _aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT;
  };

}  // namespace DO::Shakti::Vulkan
