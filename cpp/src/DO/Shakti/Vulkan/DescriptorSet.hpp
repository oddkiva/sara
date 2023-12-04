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
    class Builder;
    friend class Builder;

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

  private:
    VkDevice _device = nullptr;
    VkDescriptorSetLayout _handle = nullptr;
  };

  class DescriptorSetLayout::Builder
  {
  public:
    explicit Builder(VkDevice device)
      : _device{device}
    {
    }

    auto push_uniform_buffer_layout_binding(const std::uint32_t binding)
        -> DescriptorSetLayout::Builder&
    {
      // UBO object: matrix-view-projection matrix stack
      auto ubo_layout_binding = VkDescriptorSetLayoutBinding{};

      // In the vertex shader code, we have something like:
      // layout(binding = 0) uniform UBO { ... } ubo;
      ubo_layout_binding.binding = binding;
      ubo_layout_binding.descriptorCount = 1;  // TODO: see if this ever
                                               // needs to change.
      ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      ubo_layout_binding.pImmutableSamplers = nullptr;

      // Accessible from the vertex shader only.
      ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

      _bindings.emplace_back(ubo_layout_binding);

      return *this;
    }

    auto push_image_sampler_layout_binding(const std::uint32_t binding)
        -> DescriptorSetLayout::Builder&
    {
      auto sampler_layout_binding = VkDescriptorSetLayoutBinding{};

      sampler_layout_binding.binding = binding;
      sampler_layout_binding.descriptorCount = 1;  // TODO: see if this ever
                                                   // needs to change.
      sampler_layout_binding.descriptorType =
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      sampler_layout_binding.pImmutableSamplers = nullptr;

      // Accessible from the fragment shader only.
      sampler_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

      _bindings.emplace_back(sampler_layout_binding);

      return *this;
    }

    auto create() const -> DescriptorSetLayout
    {
      auto create_info = VkDescriptorSetLayoutCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      create_info.bindingCount = static_cast<std::uint32_t>(_bindings.size());
      create_info.pBindings = _bindings.data();

      auto desc_set_layout = DescriptorSetLayout{};
      desc_set_layout._device = _device;
      const auto status = vkCreateDescriptorSetLayout(  //
          _device, &create_info, nullptr,               //
          &desc_set_layout._handle                      //
      );
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Error: failed to create descriptor set layout! "
                        "Error code: {}",
                        static_cast<int>(status))};

      return desc_set_layout;
    }

  private:
    VkDevice _device = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayoutBinding> _bindings;
  };


  // According to:
  // https://arm-software.github.io/vulkan_best_practice_for_mobile_developers/samples/performance/descriptor_management/descriptor_management_tutorial.html
  // We don't need to free descriptor sets manually, so `vkFreeDescriptorSets`
  // is not needed, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT is not
  // needed.
  class DescriptorSets
  {
  public:
    DescriptorSets() = default;

    DescriptorSets(const DescriptorSets&) = delete;

    DescriptorSets(DescriptorSets&& other)
    {
      swap(other);
    }

    DescriptorSets(const VkDescriptorSetLayout* descriptor_set_layouts,
                   const std::uint32_t count,
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

#if 0
    ~DescriptorSets()
    {
      if (_device == nullptr || _pool == nullptr || _handles.empty())
        return;
      vkFreeDescriptorSets(_device, _pool,
                           static_cast<std::uint32_t>(_handles.size()),
                           _handles.data());
      _handles.clear();
    }
#endif

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

    auto operator[](const std::uint32_t i) -> VkDescriptorSet&
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
