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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <vulkan/vulkan_core.h>

#include <fmt/core.h>

#include <vector>


namespace DO::Shakti::Vulkan {

  class DescriptorPool;
  class DescriptorSet;

  class DescriptorPool
  {
    friend class DescriptorSet;

  public:
    class Builder;
    friend class Builder;

  public:
    DescriptorPool() = default;

    DescriptorPool(const DescriptorPool&) = delete;

    DescriptorPool(DescriptorPool&& other)
    {
      swap(other);
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


  class DescriptorPool::Builder
  {
  public:
    Builder() = delete;

    Builder(const VkDevice device)
      : _device{device}
    {
      _create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    }

    auto pool_count(const std::uint32_t n) -> Builder&
    {
      _pool_sizes.resize(n);
      _create_info.poolSizeCount =
          static_cast<std::uint32_t>(_pool_sizes.size());
      _create_info.pPoolSizes = _pool_sizes.data();
      return *this;
    }

    auto pool_max_sets(const std::uint32_t n) -> Builder&
    {
      _create_info.maxSets = n;
      return *this;
    }

    auto pool_type(const std::uint32_t i) -> VkDescriptorType&
    {
      return _pool_sizes[i].type;
    }

    auto descriptor_count(const std::uint32_t i) -> std::uint32_t&
    {
      return _pool_sizes[i].descriptorCount;
    }

    auto create() -> DescriptorPool
    {
      auto pool = DescriptorPool{};
      pool._device = _device;

      _create_info.poolSizeCount = _pool_sizes.size();
      _create_info.pPoolSizes = _pool_sizes.data();
      _create_info.maxSets = static_cast<uint32_t>(1);

      const auto status = vkCreateDescriptorPool(_device, &_create_info,
                                                 nullptr, &pool._handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{fmt::format(
            "[VK] Error: failed to create descriptor pool! Error code: {}",
            static_cast<int>(status))};

      return pool;
    }

  private:
    VkDevice _device = nullptr;

    std::vector<VkDescriptorPoolSize> _pool_sizes;
    VkDescriptorPoolCreateInfo _create_info = {};
  };

}  // namespace DO::Shakti::Vulkan
