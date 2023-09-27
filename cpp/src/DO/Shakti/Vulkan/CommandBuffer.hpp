#pragma once

#include <DO/Shakti/Vulkan/CommandPool.hpp>
#include <DO/Shakti/Vulkan/Device.hpp>
#include <DO/Shakti/Vulkan/Fence.hpp>
#include <DO/Shakti/Vulkan/Framebuffer.hpp>
#include <DO/Shakti/Vulkan/GraphicsPipeline.hpp>
#include <DO/Shakti/Vulkan/RenderPass.hpp>
#include <DO/Shakti/Vulkan/Swapchain.hpp>


namespace DO::Shakti::Vulkan {

  class CommandBufferSequence
  {
  public:
    CommandBufferSequence() = default;

    CommandBufferSequence(const std::uint32_t num_buffers,
                          const VkDevice device,
                          const VkCommandPool command_pool)
      : _device{device}
      , _command_pool{command_pool}
    {
      auto alloc_info = VkCommandBufferAllocateInfo{};
      alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      alloc_info.commandPool = _command_pool;
      alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      alloc_info.commandBufferCount = num_buffers;

      _command_buffers.resize(num_buffers);
      const auto status = vkAllocateCommandBuffers(_device, &alloc_info,
                                                   _command_buffers.data());
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("Failed to allocate command buffers! Error code: {}",
                        static_cast<int>(status))};
    }

    CommandBufferSequence(const CommandBufferSequence&) = delete;

    CommandBufferSequence(CommandBufferSequence&& other)
    {
      swap(other);
    }

    ~CommandBufferSequence()
    {
      clear();
    }

    auto operator=(const CommandBufferSequence&)
        -> CommandBufferSequence& = delete;

    auto operator=(CommandBufferSequence&& other) -> CommandBufferSequence&
    {
      swap(other);
      return *this;
    }

    auto swap(CommandBufferSequence& other) -> void
    {
      std::swap(_device, other._device);
      std::swap(_command_pool, other._command_pool);
      std::swap(_command_buffers, other._command_buffers);
    }

    auto clear() -> void
    {
      if (_device == nullptr || _command_pool == nullptr)
        return;

      SARA_DEBUG << fmt::format(
          "[VK] Freeing command buffers: [ptr:{}] [size:{}]\n",
          fmt::ptr(_command_buffers.data()), _command_buffers.size());
      vkFreeCommandBuffers(_device, _command_pool,
                           static_cast<std::uint32_t>(_command_buffers.size()),
                           _command_buffers.data());
      _command_buffers.clear();
    }

    auto size() const -> std::size_t
    {
      return _command_buffers.size();
    }

    auto operator[](const int i) -> VkCommandBuffer&
    {
      return _command_buffers[i];
    }

    auto operator[](const int i) const -> VkCommandBuffer
    {
      return _command_buffers[i];
    }

    auto reset(int i, VkCommandBufferResetFlags flags = 0) const -> void
    {
      vkResetCommandBuffer(_command_buffers[i], flags);
    }

    auto data() const -> const VkCommandBuffer*
    {
      return _command_buffers.data();
    }

  private:
    VkDevice _device = nullptr;
    VkCommandPool _command_pool = nullptr;
    std::vector<VkCommandBuffer> _command_buffers;
  };

}  // namespace DO::Shakti::Vulkan
