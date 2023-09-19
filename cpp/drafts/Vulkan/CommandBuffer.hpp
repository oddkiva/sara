#pragma once

#include <drafts/Vulkan/CommandPool.hpp>
#include <drafts/Vulkan/Device.hpp>
#include <drafts/Vulkan/Fence.hpp>
#include <drafts/Vulkan/Framebuffer.hpp>
#include <drafts/Vulkan/GraphicsPipeline.hpp>
#include <drafts/Vulkan/Queue.hpp>
#include <drafts/Vulkan/RenderPass.hpp>
#include <drafts/Vulkan/Swapchain.hpp>


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

    auto swap(CommandBufferSequence& other) -> void
    {
      std::swap(_device, other._device);
      std::swap(_command_pool, other._command_pool);
      std::swap(_command_buffers, other._command_buffers);
    }

    auto clear() -> void
    {
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

  private:
    VkDevice _device = nullptr;
    VkCommandPool _command_pool = nullptr;
    std::vector<VkCommandBuffer> _command_buffers;
  };

}  // namespace DO::Shakti::Vulkan


namespace DO::Kalpana::Vulkan {

  inline auto record_draw_graphics_command_buffers(
      Shakti::Vulkan::CommandBufferSequence& command_buffers,
      const RenderPass& render_pass,  //
      const Swapchain& swapchain,     //
      const FramebufferSequence& framebuffers,
      const GraphicsPipeline& graphics_pipeline) -> void
  {
    for (auto i = 0u; i < command_buffers.size(); ++i)
    {
      auto begin_info = VkCommandBufferBeginInfo{};
      begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin_info.flags = 0;
      begin_info.pInheritanceInfo = nullptr;

      auto status = VkResult{};
      status = vkBeginCommandBuffer(command_buffers[i], &begin_info);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Error: failed to begin recording command buffer! "
                        "Error code: {}",
                        static_cast<int>(status))};

      auto render_pass_begin_info = VkRenderPassBeginInfo{};
      {
        render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_begin_info.renderPass = render_pass.handle;
        render_pass_begin_info.framebuffer = framebuffers[i];
        render_pass_begin_info.renderArea.offset = {0, 0};
        render_pass_begin_info.renderArea.extent = swapchain.extent;

        render_pass_begin_info.clearValueCount = 1;

        static constexpr auto clear_white_color =
            VkClearValue{{{0.f, 0.f, 0.f, 1.f}}};
        render_pass_begin_info.pClearValues = &clear_white_color;
      }

      vkCmdBeginRenderPass(command_buffers[i], &render_pass_begin_info,
                           VK_SUBPASS_CONTENTS_INLINE);
      {
        vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                          graphics_pipeline);
        vkCmdDraw(command_buffers[i], 3, 1, 0, 0);
      }
      vkCmdEndRenderPass(command_buffers[i]);

      status = vkEndCommandBuffer(command_buffers[i]);
      if (status != VK_SUCCESS)
        throw std::runtime_error{fmt::format(
            "[VK] Error: failed to end record command buffer {}! Error "
            "code: {}",  //
            i, static_cast<int>(status))};
    }
  }

}  // namespace DO::Kalpana::Vulkan
