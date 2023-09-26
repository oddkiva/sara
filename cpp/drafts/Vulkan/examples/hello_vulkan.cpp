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

#define GLFW_INCLUDE_VULKAN

#include <drafts/Vulkan/Buffer.hpp>
#include <drafts/Vulkan/CommandBuffer.hpp>
#include <drafts/Vulkan/DeviceMemory.hpp>
#include <drafts/Vulkan/EasyGLFW.hpp>
#include <drafts/Vulkan/Geometry.hpp>
#include <drafts/Vulkan/GraphicsBackend.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <signal.h>

#include <atomic>
#include <limits>
#include <stdexcept>


namespace glfw = DO::Kalpana::GLFW;
namespace kvk = DO::Kalpana::Vulkan;
namespace svk = DO::Shakti::Vulkan;


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


static const auto vertices = std::vector<Vertex>{
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},  //
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},   //
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},    //
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}    //
};

static const auto indices = std::vector<uint16_t>{
    0, 1, 2,  //
    2, 3, 0   //
};


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
    transfer_vertex_data_to_vulkan(vertices);
    transfer_element_data_to_vulkan(indices);
  }

  auto transfer_vertex_data_to_vulkan(const std::vector<Vertex>& vertices)
      -> void
  {
    const auto buffer_factory = svk::BufferFactory{_device};
    const auto dmem_factory =
        svk::DeviceMemoryFactory{_physical_device, _device};

    // Staging buffer
    SARA_DEBUG << "Vertex staging buffer...\n";
    const auto vbo_staging =
        buffer_factory.make_staging_buffer<Vertex>(vertices.size());
    const auto vdm_staging =
        dmem_factory.allocate_for_staging_buffer(vbo_staging);
    vbo_staging.bind(vdm_staging, 0);

    // Copy the data.
    SARA_DEBUG << "Copying from host data to vertex staging buffer...\n";
    vdm_staging.copy_from(vertices.data(), vertices.size(), 0);

    // Device buffer
    SARA_DEBUG << "Vertex device buffer...\n";
    _vbo = buffer_factory.make_device_vertex_buffer<Vertex>(vertices.size());
    _vdm = dmem_factory.allocate_for_device_buffer(_vbo);
    _vbo.bind(_vdm, 0);

    SARA_DEBUG << "Recording data transfer from vertex staging buffer to "
                  "device buffer...\n";
    const auto copy_cmd_bufs =
        svk::CommandBufferSequence{1, _device, _graphics_cmd_pool};
    const auto& copy_cmd_buf = copy_cmd_bufs[0];
    // Copy from the staging buffer to the device buffer
    svk::record_copy_buffer(vbo_staging, _vbo, copy_cmd_buf);

    SARA_DEBUG << "Submitting data transfer command...\n";
    _graphics_queue.submit_copy_commands(copy_cmd_bufs);
    _graphics_queue.wait();
  }

  auto
  transfer_element_data_to_vulkan(const std::vector<std::uint16_t>& indices)
      -> void
  {
    const auto buffer_factory = svk::BufferFactory{_device};
    const auto dmem_factory =
        svk::DeviceMemoryFactory{_physical_device, _device};

    // Staging buffer
    SARA_DEBUG << "Element staging buffer...\n";
    const auto ebo_staging =
        buffer_factory.make_staging_buffer<std::uint16_t>(indices.size());
    const auto edm_staging =
        dmem_factory.allocate_for_staging_buffer(ebo_staging);
    ebo_staging.bind(edm_staging, 0);

    // Copy the data.
    SARA_DEBUG << "Copying from host data to index staging buffer...\n";
    edm_staging.copy_from(indices.data(), indices.size(), 0);

    // Device buffer
    SARA_DEBUG << "Element device buffer...\n";
    _ebo =
        buffer_factory.make_device_index_buffer<std::uint16_t>(indices.size());
    _edm = dmem_factory.allocate_for_device_buffer(_ebo);
    _ebo.bind(_edm, 0);

    SARA_DEBUG << "Recording data transfer from index staging buffer to "
                  "device buffer...\n";
    const auto copy_cmd_bufs =
        svk::CommandBufferSequence{1, _device, _graphics_cmd_pool};
    const auto& copy_cmd_buf = copy_cmd_bufs[0];
    // Copy from the staging buffer to the device buffer
    svk::record_copy_buffer(ebo_staging, _ebo, copy_cmd_buf);

    SARA_DEBUG << "Submitting data transfer command...\n";
    _graphics_queue.submit_copy_commands(copy_cmd_bufs);
    _graphics_queue.wait();
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
        _device,                     //
        _swapchain.handle,
        forever,                                      // timeout in nanoseconds
        _image_available_semaphores[_current_frame],  // semaphore to signal
        VK_NULL_HANDLE,                               // fence to signal
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
    const auto wait_semaphores = std::array<VkSemaphore, 1>{
        _image_available_semaphores[_current_frame]  //
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
    const auto render_finished_semaphores = std::array<VkSemaphore, 1>{
        _render_finished_semaphores[_current_frame]  //
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

    result = vkQueuePresentKHR(_present_queue, &present_info);
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

  auto loop(GLFWwindow* window) -> void
  {
    while (!glfwWindowShouldClose(window))
    {
      glfwPollEvents();
      draw_frame();

      if (SignalHandler::ctrl_c_hit)
        break;
    }

    vkDeviceWaitIdle(_device);
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

      static const auto vbos = std::array<VkBuffer, 1>{_vbo};
      static constexpr auto offsets = std::array<VkDeviceSize, 1>{0};
      vkCmdBindVertexBuffers(command_buffer, 0,
                             static_cast<std::uint32_t>(vbos.size()),
                             vbos.data(), offsets.data());

      vkCmdBindIndexBuffer(command_buffer, _ebo, 0, VK_INDEX_TYPE_UINT16);

      vkCmdDrawIndexed(command_buffer,
                       static_cast<std::uint32_t>(indices.size()), 1, 0, 0, 0);
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

  svk::Buffer _vbo;
  svk::Buffer _ebo;

  svk::DeviceMemory _vdm;
  svk::DeviceMemory _edm;
};


int main(int, char**)
{
  SARA_CHECK(vertices.size());
  SARA_CHECK(sizeof(Vertex));
  SARA_CHECK(vertices.size() * sizeof(Vertex));

  // return 0;

  SignalHandler::init();

  auto app = glfw::Application{};
  app.init_for_vulkan_rendering();

  const auto app_name = "Vulkan Triangle";
  auto window = glfw::Window{300, 300, app_name};

  try
  {
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
