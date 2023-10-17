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

#include <DO/Shakti/Vulkan/PhysicalDevice.hpp>

#include <fmt/core.h>

#include <utility>


namespace DO::Shakti::Vulkan {

  class Sampler
  {
  public:
    class Builder;
    friend class Builder;

    Sampler() = default;

    Sampler(const Sampler&) = delete;

    Sampler(Sampler&& other) = default;

    ~Sampler()
    {
      if (_device == VK_NULL_HANDLE || _handle == VK_NULL_HANDLE)
        return;
      vkDestroySampler(_device, _handle, nullptr);
    }

    auto operator=(const Sampler&) -> Sampler& = delete;

    auto operator=(Sampler&& other) -> Sampler&
    {
      _device = std::move(other._device);
      _handle = std::move(other._handle);
      return *this;
    }

    operator VkSampler&()
    {
      return _handle;
    }

    operator VkSampler() const
    {
      return _handle;
    }

  private:
    VkDevice _device = VK_NULL_HANDLE;
    VkSampler _handle = VK_NULL_HANDLE;
  };


  class Sampler::Builder
  {
  public:
    Builder(const PhysicalDevice& physical_device, const VkDevice device)
      : _physical_device{physical_device}
      , _device{device}
    {
    }

    auto create() -> Sampler
    {
      auto properties = _physical_device.properties();

      auto create_info = VkSamplerCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      // Bilinear interpolation.
      create_info.magFilter = VK_FILTER_LINEAR;
      create_info.minFilter = VK_FILTER_LINEAR;
      // For out-of-bound sampling.
      create_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      create_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      create_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      // Anisotropy.
      create_info.anisotropyEnable = VK_TRUE;
      create_info.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
      //
      create_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
      create_info.unnormalizedCoordinates = VK_FALSE;
      create_info.compareEnable = VK_FALSE;
      create_info.compareOp = VK_COMPARE_OP_ALWAYS;
      create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

      auto sampler = Sampler{};
      sampler._device = _device;
      const auto status = vkCreateSampler(_device, &create_info, nullptr,  //
                                          &sampler._handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Failed to create sampler! Error code: {}",
                        static_cast<int>(status))};

      return sampler;
    }

  private:
    const PhysicalDevice& _physical_device;
    VkDevice _device = VK_NULL_HANDLE;
  };

}  // namespace DO::Shakti::Vulkan
