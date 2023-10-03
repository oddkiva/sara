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

#include <DO/Shakti/Vulkan/DescriptorPool.hpp>

#include <fmt/core.h>

#include <vector>


namespace DO::Shakti::Vulkan {

  class DescriptorSetLayout
  {
  public:
    DescriptorSetLayout() = default;

    DescriptorSetLayout(const DescriptorSetLayout&) = delete;

    DescriptorSetLayout(DescriptorSetLayout&& other)
    {
      swap(other);
    }

    ~DescriptorSetLayout()
    {
      if (_device == nullptr || _handle == nullptr)
        return;
      vkDestroyDescriptorSetLayout(_device, _handle, nullptr);
    }

    auto operator=(const DescriptorSetLayout&) -> DescriptorSetLayout& = delete;

    auto operator=(DescriptorSetLayout&& other) -> DescriptorSetLayout&
    {
      swap(other);
      return *this;
    }

    operator VkDescriptorSetLayout&()
    {
      return _handle;
    }

    operator VkDescriptorSetLayout() const
    {
      return _handle;
    }

    auto swap(DescriptorSetLayout& other) -> void
    {
      std::swap(_device, other._device);
      std::swap(_handle, other._handle);
    }

    static auto create_for_single_ubo(VkDevice device) -> DescriptorSetLayout
    {
      // UBO object: matrix-view-projection matrix stack
      auto ubo_layout_binding = VkDescriptorSetLayoutBinding{};
      ubo_layout_binding.binding = 0;
      ubo_layout_binding.descriptorCount = 1;
      ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      ubo_layout_binding.pImmutableSamplers = nullptr;
      ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

      // We only need 1 set of descriptors for the MVP UBO.
      auto create_info = VkDescriptorSetLayoutCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      create_info.bindingCount = 1;
      create_info.pBindings = &ubo_layout_binding;

      // Finally really create the descriptor set layout.
      auto ubo_set_layout = DescriptorSetLayout{};
      ubo_set_layout._device = device;
      const auto status = vkCreateDescriptorSetLayout(
          device, &create_info, nullptr, &ubo_set_layout._handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{fmt::format(
            "[VK] Error: failed to create UBO set layout! Error code: {}",
            static_cast<int>(status))};

      return ubo_set_layout;
    }

  private:
    VkDevice _device = nullptr;
  public:
    VkDescriptorSetLayout _handle = nullptr;
  };


  class DescriptorSets
  {
  public:
    DescriptorSets() = default;

    DescriptorSets(const DescriptorSets&) = delete;

    DescriptorSets(DescriptorSets&& other)
    {
      swap(other);
    }

    DescriptorSets(const std::uint32_t count,
                   const VkDescriptorSetLayout* descriptor_set_layouts,
                   const DescriptorPool& descriptor_pool)
      : _device{descriptor_pool._device}
      , _pool{descriptor_pool._handle}
    {
      _handles.resize(count);

      auto allocate_info = VkDescriptorSetAllocateInfo{};
      allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      allocate_info.descriptorPool = _pool;
      allocate_info.descriptorSetCount = count;
      allocate_info.pSetLayouts = descriptor_set_layouts;

      const auto status =
          vkAllocateDescriptorSets(_device, &allocate_info, _handles.data());
      if (status != VK_SUCCESS)
        throw std::runtime_error{fmt::format(
            "[VK] Error: failed to create descriptor pool! Error code: {}",
            static_cast<int>(status))};
    }

    ~DescriptorSets()
    {
      if (_device == nullptr || _pool == nullptr || _handles.empty())
        return;

      vkFreeDescriptorSets(_device, _pool,
                           static_cast<std::uint32_t>(_handles.size()),
                           _handles.data());
      _handles.clear();
    }

    auto operator=(const DescriptorSets&) -> DescriptorSets& = delete;

    auto operator=(DescriptorSets&& other) -> DescriptorSets&
    {
      swap(other);
      return *this;
    }

    auto operator[](const std::uint32_t i) const -> VkDescriptorSet
    {
      return _handles[i];
    }

    auto swap(DescriptorSets& other) -> void
    {
      std::swap(_device, other._device);
      std::swap(_pool, other._pool);
      _handles.swap(other._handles);
    }

  private:
    VkDevice _device = nullptr;
    VkDescriptorPool _pool = nullptr;
    std::vector<VkDescriptorSet> _handles;
  };

}  // namespace DO::Shakti::Vulkan
