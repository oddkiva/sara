#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN

#include <drafts/Vulkan/EasyGLFW.hpp>
#include <drafts/Vulkan/GraphicsBackend.hpp>

#include <signal.h>

#include <atomic>
#include <limits>
#include <stdexcept>


namespace glfw = DO::Kalpana::GLFW;
namespace kvk = DO::Kalpana::Vulkan;


struct SignalHandler
{
  static bool initialized;
  static std::atomic_bool ctrl_c_hit;
  static struct sigaction sigint_handler;

  static auto stop_render_loop(int) -> void
  {
    std::cout << "[CTRL+C HIT] Preparing to close the program!" << std::endl;
    ctrl_c_hit = true;
  }

  static auto init() -> void
  {
    if (initialized)
      return;

#if defined(_WIN32)
    signal(SIGINT, stop_render_loop);
    signal(SIGTERM, stop_render_loop);
    signal(SIGABRT, stop_render_loop);
#else
    sigint_handler.sa_handler = SignalHandler::stop_render_loop;
    sigemptyset(&sigint_handler.sa_mask);
    sigint_handler.sa_flags = 0;
    sigaction(SIGINT, &sigint_handler, nullptr);
#endif

    initialized = true;
  }
};

bool SignalHandler::initialized = false;
std::atomic_bool SignalHandler::ctrl_c_hit = false;
#if !defined(_WIN32)
struct sigaction SignalHandler::sigint_handler = {};
#endif


static auto get_program_path() -> std::filesystem::path
{
#ifdef _WIN32
  static auto path = std::array<wchar_t, MAX_PATH>{};
  GetModuleFileNameW(nullptr, path.data(), MAX_PATH);
  return path.data();
#else
  static auto result = std::array<char, PATH_MAX>{};
  ssize_t count = readlink("/proc/self/exe", result.data(), PATH_MAX);
  return std::string(result.data(), (count > 0) ? count : 0);
#endif
}


class VulkanTriangleRenderer : public kvk::GraphicsBackend
{
public:
  VulkanTriangleRenderer(GLFWwindow* window, const std::string& app_name,
                         const std::filesystem::path& shader_dirpath,
                         const bool debug_vulkan = true)
    : kvk::GraphicsBackend{window, app_name,             //
                           shader_dirpath / "vert.spv",  //
                           shader_dirpath / "frag.spv",  //
                           debug_vulkan}
  {
  }

  auto draw_frame() -> void
  {
    static constexpr auto forever = std::numeric_limits<std::uint64_t>::max();
    auto result = VkResult{};

    SARA_CHECK(_current_frame);

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
    SARA_DEBUG << "[VK] Waiting for the render fence to signal...\n";
    _render_fences[_current_frame].wait(forever);
    // This function call blocks.
    //
    // After that, the function unblocks and the program resumes its execution
    // flow on the CPU side.

    // Acquire the next image ready to be rendered.
    SARA_DEBUG << "[VK] Acquiring the next image ready to be rendered...\n";
    auto index_of_next_image_to_render = std::uint32_t{};
    result = vkAcquireNextImageKHR(  //
        _device.handle,              //
        _swapchain.handle,
        forever,  // timeout in nanoseconds
        _image_available_semaphores[_current_frame]
            ._handle,    // semaphore to signal
        VK_NULL_HANDLE,  // fence to signal
        &index_of_next_image_to_render);
    // Sanity check.
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
      throw std::runtime_error("failed to acquire the next swapchain image!");
#if defined(RECREATE_SWAPCHAIN_IMPLEMENTED)
    // Recreate the swapchain if the size of the window surface has changed.
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
      recreate_swapchain();
      return;
    }
#endif
    SARA_CHECK(index_of_next_image_to_render);

    // Reset the signaled fence associated to the current frame to an unsignaled
    // state. So that the GPU can reuse it to signal.
    SARA_DEBUG << "[VK] Resetting for the render fence...\n";
    _render_fences[_current_frame].reset();

    // Reset the command buffer associated to the current frame.
    SARA_DEBUG << "[VK] Resetting for the command buffer...\n";
    vkResetCommandBuffer(_graphics_cmd_bufs[_current_frame],
                         /*VkCommandBufferResetFlagBits*/ 0);

    // Record the draw command to be performed on this swapchain image.
    SARA_CHECK(_framebuffers.fbs.size());
    record_graphics_command_buffer(
        _graphics_cmd_bufs[_current_frame],
        _framebuffers[index_of_next_image_to_render]);

    // Submit the draw command to the graphics queue.
    SARA_DEBUG << "[VK] Specifying the graphics command submission...\n";
    auto submit_info = VkSubmitInfo{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // Specify the draw command buffer to submit and the dependency order in
    // which the draw command happens.
    //
    // A. The easy bit: reference the draw command buffer we want to submit.
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &_graphics_cmd_bufs[_current_frame];

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
        _image_available_semaphores[_current_frame]._handle  //
    };
    submit_info.waitSemaphoreCount =
        static_cast<std::uint32_t>(wait_semaphores.size());
    submit_info.pWaitSemaphores = wait_semaphores.data();

    // 2. Ensure that drawing starts until the GPU has finished processing this
    //    image. (Worry about this later).
    static constexpr auto wait_stages = std::array<VkPipelineStageFlags, 1>{
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submit_info.pWaitDstStageMask = wait_stages.data();

    // 3. Set the render finished semaphore in a signaled state, when the draw
    //    command completes.
    //    This is for the present command buffer.
    const auto render_finished_semaphores = std::array{
        _render_finished_semaphores[_current_frame]._handle  //
    };
    submit_info.signalSemaphoreCount =
        static_cast<std::uint32_t>(render_finished_semaphores.size());
    submit_info.pSignalSemaphores = render_finished_semaphores.data();

    // 4. Submit the draw command.
    //    - Notice the fence parameter passed to vkQueueSubmit.
    //
    //    - It has been reset by vkResetFences above so that it can be in
    //      signaled state when the draw command completes.
    //
    //    - When we re-invoke the `draw_frame` command, and this draw_frame
    //      needs to reuse the same swapchain image, i.e., the one with the same
    //      index `_current_frame`,
    //
    //      the first command `vkWaitForFences(...)` at the beginning of the
    //      draw command stalls the CPU execution flow, until the current draw
    //      command submission, here, completes.
    //
    //      After which, the fence `_render_fences[_current_frame]` enters in a
    //      signaled state and un-stalls the function `vkWaitForFences(...)`.
    _graphics_queue.submit(submit_info, _render_fences[_current_frame]);

    // Submit the present command to the present queue.
    auto present_info = VkPresentInfoKHR{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    // Wait until the draw command finishes. It is signaled by its signal
    // semaphore.
    present_info.waitSemaphoreCount =
        static_cast<std::uint32_t>(render_finished_semaphores.size());
    present_info.pWaitSemaphores = render_finished_semaphores.data();
    // Specify which swapchain we are using.
    VkSwapchainKHR swapchains[] = {_swapchain.handle};
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swapchains;
    present_info.pImageIndices = &index_of_next_image_to_render;

    result = vkQueuePresentKHR(_present_queue.handle, &present_info);
    if (result != VK_SUCCESS)
      throw std::runtime_error{fmt::format(
          "failed to present the swapchain image {}!", _current_frame)};
#if 0
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
        framebufferResized)
    {
      framebuffer_resized = false;
      recreate_swapchain();
    }
#endif

    // Update the current frame to the next one for the next `draw_frame` call.
    const auto max_frames_in_flight = _swapchain.images.size();
    _current_frame = (_current_frame + 1) % max_frames_in_flight;
  }

  auto loop(GLFWwindow* window)
  {
    while (!glfwWindowShouldClose(window))
    {
      glfwPollEvents();
      draw_frame();

      if (SignalHandler::ctrl_c_hit)
        break;
    }

    vkDeviceWaitIdle(_device.handle);
  }

  auto record_graphics_command_buffer(VkCommandBuffer command_buffer,
                                      VkFramebuffer framebuffer) -> void
  {
    SARA_DEBUG << "[VK] Recording graphics command buffer...\n";
    auto begin_info = VkCommandBufferBeginInfo{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = 0;
    begin_info.pInheritanceInfo = nullptr;

    auto status = VkResult{};
    status = vkBeginCommandBuffer(command_buffer, &begin_info);
    if (status != VK_SUCCESS)
      throw std::runtime_error{
          fmt::format("[VK] Error: failed to begin recording command buffer! "
                      "Error code: {}",
                      static_cast<int>(status))};

    auto render_pass_begin_info = VkRenderPassBeginInfo{};
    {
      render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      render_pass_begin_info.renderPass = _render_pass.handle;
      render_pass_begin_info.framebuffer = framebuffer;
      render_pass_begin_info.renderArea.offset = {0, 0};
      render_pass_begin_info.renderArea.extent = _swapchain.extent;

      render_pass_begin_info.clearValueCount = 1;

      static constexpr auto clear_white_color =
          VkClearValue{{{0.f, 0.f, 0.f, 1.f}}};
      render_pass_begin_info.pClearValues = &clear_white_color;
    }

    SARA_DEBUG << "[VK] Begin render pass...\n";
    vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info,
                         VK_SUBPASS_CONTENTS_INLINE);
    {
      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        _graphics_pipeline);

#ifdef ALLOW_DYNAMIC_VIEWPORT_AND_SCISSOR_STATE
      VkViewport viewport{};
      viewport.x = 0.0f;
      viewport.y = 0.0f;
      viewport.width = static_cast<float>(_swapchain.extent.width);
      viewport.height = static_cast<float>(_swapchain.extent.height);
      viewport.minDepth = 0.0f;
      viewport.maxDepth = 1.0f;
      vkCmdSetViewport(command_buffer, 0, 1, &viewport);

      VkRect2D scissor{};
      scissor.offset = {0, 0};
      scissor.extent = _swapchain.extent;
      vkCmdSetScissor(command_buffer, 0, 1, &scissor);
#endif

      vkCmdDraw(command_buffer, 3, 1, 0, 0);
    }

    SARA_DEBUG << "[VK] End render pass...\n";
    vkCmdEndRenderPass(command_buffer);

    status = vkEndCommandBuffer(command_buffer);
    if (status != VK_SUCCESS)
      throw std::runtime_error{fmt::format(
          "[VK] Error: failed to end record command buffer! Error code: {}",
          static_cast<int>(status))};
  }

private:
  int _current_frame = 0;
};


int main(int, char**)
{
  SignalHandler::init();

  auto app = glfw::Application{};
  app.init_for_vulkan_rendering();

  const auto app_name = "Vulkan Triangle";
  auto window = glfw::Window{300, 300, app_name};

  try
  {
    namespace fs = std::filesystem;

    const auto program_dir_path = get_program_path().parent_path();
    auto triangle_renderer = VulkanTriangleRenderer{
        window, app_name, program_dir_path / "hello_vulkan_shaders", true};
    triangle_renderer.loop(window);
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
