class VulkanBackend
{
private: /* initialization methods */
private: /* cleanup methods related to the render pipeline. */
  // auto cleanup_render_pipeline() -> void {
  //   vkFreeCommandBuffers(_device, commandPool,
  //                        static_cast<uint32_t>(commandBuffers.size()),
  //                        commandBuffers.data());
  //   vkDestroyPipeline(_device, _graphics_pipeline, nullptr);
  //   vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
  //   vkDestroyRenderPass(_device, _render_pass, nullptr);
  // }

private: /* cleanup methods related to the present operations */
};


class HelloTriangleApplication
{
public:
  auto run() -> void
  {
    init();
    cleanup();
  }

  auto draw_frame() -> void
  {
    // The GPU is blocking the CPU at this point of the road:
    // - The CPU is stopped at a fence
    // - The first fence is closed
    vkWaitForFences(_device, 1, &_in_flight_fences[current_frame], VK_TRUE,
                    UINT64_MAX);
    // - The first fence has opened just now.

    // The CPU pursues its journey:
    // - it acquires the next image ready for presentation.
    auto image_index = std::uint32_t{};
    auto result =
        vkAcquireNextImageKHR(_device, _swapchain, UINT64_MAX,
                              _image_available_semaphores[current_frame],
                              VK_NULL_HANDLE, &image_index);
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
      // For example when the window is resized, we need to recreate the
      // swapchain, the images of the swap chains must reinitialized with the
      // new window extent.
      recreate_swapchain();
      return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
      throw std::runtime_error("failed to acquire swapchain image!");

    // The CPU encounters another fence controlled by the GPU:
    // - The CPU is waiting from the GPU until the image becomes available
    //   for rendering.
    // - The second fence displays a red light
    // - When rendering for the first time, the second fence is already
    // opened,
    //   so the CPU can pursue its journey.
    if (_images_in_flight[image_index] != VK_NULL_HANDLE)
      vkWaitForFences(_device, 1, &_images_in_flight[current_frame], VK_TRUE,
                      UINT64_MAX);
    _images_in_flight[image_index] = _in_flight_fences[current_frame];
    // - The second fence has opened just now.

    // The CPU pursues its journey:
    //
    // - The CPU will now detail what to do at the drawing stage.
    auto submit_info = VkSubmitInfo{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkPipelineStageFlags wait_stages =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // - The CPU tells the GPU to ensure that it starts drawing only when the
    // image becomes available.
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &_image_available_semaphores[current_frame];
    submit_info.pWaitDstStageMask = &wait_stages;

    // - The CPU tells the GPU to ensure that it notifies when the drawing is
    //   finished and thus ready to present onto the screen.
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &_render_finished_semaphores[current_frame];

    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &_command_buffers[image_index];

    // - The CPU now tells the GPU to make the first fence close the road.
    vkResetFences(_device, 1, &_in_flight_fences[current_frame]);

    // - The CPU submits a drawing command to the GPU (on the graphics
    // queue).
    if (vkQueueSubmit(_graphics_queue, 1, &submit_info,
                      _in_flight_fences[current_frame]) != VK_SUCCESS)
      throw std::runtime_error{"Failed to submit draw command buffer!"};

    // - The CPU details the screen presentation command.
    auto present_info = VkPresentInfoKHR{};
    {
      present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

      // - The CPU tells the GPU to trigger the frame display only when the
      //   rendering is finished.
      //
      //   This completely specifies the dependency between the rendering
      //   command and the screen display command.
      present_info.waitSemaphoreCount = 1;
      present_info.pWaitSemaphores =
          &_render_finished_semaphores[current_frame];

      present_info.swapchainCount = 1;
      present_info.pSwapchains = &_swapchain;

      present_info.pImageIndices = &image_index;
    }

    // - The CPU submits another command to the GPU (on the present queue).
    vkQueuePresentKHR(_present_queue, &present_info);

    // Move to the next framebuffer.
    current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

private:
  auto init() -> void
  {
    _glfw_window.init();
    _vulkan_backend.init(_glfw_window);
  }

  auto cleanup() -> void
  {
    _vulkan_backend.cleanup();
    _glfw_window.cleanup();
  }

private:
  SingleGLFWWindowApplication _glfw_window;
  VulkanBackend _vulkan_backend;
};


int main(int, char**)
{
  auto app = HelloTriangleApplication{};
  app.run();

  return 0;
}
