#include <vulkan/vulkan.h>

#include <drafts/Vulkan/Fence.hpp>
#include <drafts/Vulkan/Semaphore.hpp>


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
    // The number of images in-flight is the number of swapchain images.
    // And there are as many fences as swapchain images.
    //
    // The pre-condition to start the render loop is to initialize our Vulkan
    // application as follows:
    // - every fence is reset to an unsignalled state. This is necessary to for
    //   the synchronization machinery to start correctly.
    // - the current frame index is 0;

    // Wait for the GPU signal that the current frame becomes available.
    //
    // The function call `vkQueueSubmit(...)` at the end of this `draw_frame`
    // method uses this `_in_flight_fences[_current_frame]` fence.
    vkWaitForFences(_device, 1, &_in_flight_fences[_current_frame], VK_TRUE,
                    UINT64_MAX);
    // This function call blocks.
    //
    // After that, the function unblocks and the program resumes its execution
    // flow on the CPU side.

    // Acquire the next image ready to be rendered.
    auto index_of_next_image_to_render = std::uint32_t{};
    const auto result = vkAcquireNextImageKHR(
        _device, _swapchain,
        UINT64_MAX,                                   // timeout in nanoseconds
        _image_available_semaphores[_current_frame],  // semaphore to signal
        VK_NULL_HANDLE,                               // fence to signal
        &index_of_next_image_to_render);
    // Sanity check.
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
      throw std::runtime_error("failed to acquire the next swapchain image!");
    // Recreate the swapchain if the size of the window surface has changed.
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
      recreate_swapchain();
      return;
    }

    // Reset the signaled fence associated to the current frame to an unsignaled
    // state. So that the GPU can reuse it to signal.
    vkResetFences(_device, 1, &_in_flight_fences[_current_frame]);

    // Reset the command buffer associated to the current frame.
    vkResetCommandBuffer(_command_buffers[_current_frame],
                         /*VkCommandBufferResetFlagBits*/ 0);
    // Record the draw command to be performed on this swapchain image.
    recordCommandBuffer(_command_buffers[_current_frame],
                        index_of_next_image_to_render);

    // Submit the draw command to the graphics queue.
    auto submit_info = VkSubmitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // Specify the draw command buffer to submit and the dependency order in
    // which the draw command happens.
    //
    // A. The easy bit: reference the draw command buffer we want to submit.
    submit_info.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

    // B. Synchronization details on the GPU side.
    //
    // 1. Wait to acquire the next image ready to be rendered.
    //
    //    When vkAcquireNextImageKHR completes, the semaphore
    //    `_image_available_semaphores[current_frame]` is in a signaled state.
    //
    //    The draw command starts only after this semaphore is in a signaled
    //    state and cannot start before.
    const auto wait_semaphores = std::array{
        _image_available_semaphores[currentFrame]  //
    };
    submitInfo.waitSemaphoreCount = wait_semaphores.size();
    submitInfo.pWaitSemaphores = wait_semaphores.data();

    // 2. Ensure that drawing starts until the GPU has finished processing this
    //    image. (Worry about this later).
    VkPipelineStageFlags wait_stages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.pWaitDstStageMask = waitStages;

    // 3. Set the render finished semaphore in a signaled state, when the draw
    //    command completes.
    //    This is for the present command buffer.
    const auto render_finished_semaphores = std::array{
        _render_finished_semaphores[_current_frame]  //
    };
    submitInfo.signalSemaphoreCount = render_finished_semaphores.size();
    submitInfo.pSignalSemaphores = render_finished_semaphores.data();

    // 4. Submit the draw command.
    //    - Notice the fence parameter passed to vkQueueSubmit.
    //    - It has been reset by vkResetFences above so that it can be in
    //      signaled state when the draw command completes.
    //    - The first command vkWaitForFences at the beginning of the draw
    //      command stalls the CPU execution flow, we need to re-render on this
    //      swapchain image.
    if (vkQueueSubmit(graphics_queue, 1, &submit_info,
                      _in_flight_fences[_current_frame]) != VK_SUCCESS)
      throw std::runtime_error("failed to submit draw command buffer!");

    // Submit the present command to the present queue.
    auto present_info = VkPresentInfoKHR{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    // Wait until the draw command finishes. It is signaled by its signal
    // semaphore.
    present_info.waitSemaphoreCount = render_finished_semaphores.size();
    present_info.pWaitSemaphores = render_finished_semaphores.data();
    // Specify which swapchain we are using.
    VkSwapchainKHR swapchains[] = {_swapchain};
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swapchains;
    present_info.pImageIndices = &index_of_next_image_to_render;

    result = vkQueuePresentKHR(present_queue, &present_info);
    if (result != VK_SUCCESS)
      throw std::runtime_error("failed to present swap chain image!");
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
        framebufferResized)
    {
      framebuffer_resized = false;
      recreate_swapchain();
    }

    // Update the current frame to the next one for the next `draw_frame` call.
    _current_frame = (_current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
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
