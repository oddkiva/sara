#pragma once

#include <DO/Shakti/Vulkan/DescriptorPool.hpp>

#include <fmt/core.h>
#include <vulkan/vulkan_core.h>


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

    static auto make_for_uniform_buffer(VkDevice device) -> DescriptorSetLayout
    {
      // UBO object: matrix-view-projection matrices
      auto ubo_layout_binding = VkDescriptorSetLayoutBinding{};
      ubo_layout_binding.binding = 0;
      ubo_layout_binding.descriptorCount = 1;
      ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      ubo_layout_binding.pImmutableSamplers = nullptr;
      ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

      auto create_info = VkDescriptorSetLayoutCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      create_info.bindingCount = 1;
      create_info.pBindings = &ubo_layout_binding;

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
    VkDescriptorSetLayout _handle = nullptr;
  };


  class DescriptorSet
  {
  public:
    DescriptorSet() = default;

    DescriptorSet(const DescriptorSet&) = delete;

    DescriptorSet(DescriptorSet&& other)
    {
      swap(other);
    }

    DescriptorSet(const std::uint32_t count,
                  const DescriptorPool& descriptor_pool)
      : _device{descriptor_pool._device}
      , _pool{descriptor_pool._handle}
      , _count{count}
    {

      auto allocate_info = VkDescriptorSetAllocateInfo{};
      allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      allocate_info.descriptorPool = _pool;
      allocate_info.descriptorSetCount = count;
      // TODO.
      allocate_info.pSetLayouts = nullptr;

      const auto status =
          vkAllocateDescriptorSets(_device, &allocate_info, nullptr, &_handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{fmt::format(
            "[VK] Error: failed to create descriptor pool! Error code: {}",
            static_cast<int>(status))};
    }

    ~DescriptorSet()
    {
      if (_device == nullptr || _pool == nullptr || _handle == nullptr)
        return;
      vkFreeDescriptorSets(_device, _pool, _count, &_handle);
    }

    auto operator=(const DescriptorSet&) -> DescriptorSet& = delete;

    auto operator=(DescriptorSet&& other) -> DescriptorSet&
    {
      swap(other);
      return *this;
    }

    operator VkDescriptorSet&()
    {
      return _handle;
    }

    operator VkDescriptorSet() const
    {
      return _handle;
    }

    auto swap(DescriptorSet& other) -> void
    {
      std::swap(_device, other._device);
      std::swap(_handle, other._handle);
    }

  private:
    VkDevice _device = nullptr;
    VkDescriptorPool _pool = nullptr;
    VkDescriptorSet _handle = nullptr;
    std::uint32_t _count = 0;
  };

}  // namespace DO::Shakti::Vulkan
