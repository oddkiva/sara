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

#include "Geometry.hpp"

#include "Common/HostUniforms.hpp"
#include "Common/SignalHandler.hpp"

#include <DO/Shakti/Vulkan/Buffer.hpp>
#include <DO/Shakti/Vulkan/CommandBuffer.hpp>
#include <DO/Shakti/Vulkan/DescriptorSet.hpp>
#include <DO/Shakti/Vulkan/DeviceMemory.hpp>
#include <DO/Shakti/Vulkan/EasyGLFW.hpp>
#include <DO/Shakti/Vulkan/GraphicsBackend.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <Eigen/Geometry>

#include <filesystem>
#include <limits>
#include <stdexcept>


namespace glfw = DO::Kalpana::GLFW;
namespace kvk = DO::Kalpana::Vulkan;
namespace svk = DO::Shakti::Vulkan;
namespace fs = std::filesystem;


// clang-format off
static const auto vertices = std::vector<Vertex>{
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{ 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{ 0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f,  0.5f}, {1.0f, 1.0f, 1.0f}}
};
// clang-format on

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
  {
    // Set the GLFW callbacks.
    {
      _window = window;
      glfwSetWindowUserPointer(window, this);
      glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);
    }

    init_instance(app_name, debug_vulkan);
    init_surface(window);
    init_physical_device();
    init_device_and_queues();
    init_swapchain(window);
    init_render_pass();
    init_swapchain_fbos();
    init_graphics_pipeline(window,  //
                           shader_dirpath / "vert.spv",
                           shader_dirpath / "frag.spv");
    init_command_pool_and_buffers();
    init_synchronization_objects();

    transfer_vertex_data_to_vulkan(vertices);
    transfer_element_data_to_vulkan(indices);

    make_ubo_descriptor_pool();
    make_ubo_descriptor_sets();
    initialize_model_view_projection_ubos();
  }

  auto init_graphics_pipeline(GLFWwindow* window,  //
                              const std::filesystem::path& vertex_shader_path,
                              const std::filesystem::path& fragment_shader_path)
      -> void override
  {
    auto w = int{};
    auto h = int{};
    glfwGetWindowSize(window, &w, &h);

    _graphics_pipeline =
        kvk::GraphicsPipeline::Builder{_device, _render_pass}
            .vertex_shader_path(vertex_shader_path)
            .fragment_shader_path(fragment_shader_path)
            .vbo_data_format<Vertex>()
            .input_assembly_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
            .viewport_sizes(static_cast<float>(w), static_cast<float>(h))
            .scissor_sizes(w, h)
            .create();
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
    _graphics_queue.submit_commands(copy_cmd_bufs);
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
    _graphics_queue.submit_commands(copy_cmd_bufs);
    _graphics_queue.wait();
  }

  auto make_ubo_descriptor_pool() -> void
  {
    const auto num_frames_in_flight =
        static_cast<std::uint32_t>(_swapchain.images.size());  // 3 typically.

    // We just need a single descriptor pool.
    auto ubo_pool_builder = svk::DescriptorPool::Builder{_device}  //
                                .pool_count(1)
                                .pool_max_sets(num_frames_in_flight);
    // This descriptor pool can only allocate UBO descriptors.
    ubo_pool_builder.pool_type(0) = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // The cumulated number of UBO descriptors across all every descriptor sets
    // cannot exceed the following number of descriptors (3).
    ubo_pool_builder.descriptor_count(0) = num_frames_in_flight;

    _ubo_pool = ubo_pool_builder.create();
  }

  auto make_ubo_descriptor_sets() -> void
  {
    // The number of frames in flight is the number of swapchain images.
    // Let's say there are 3 frames in flight.
    //
    // We will construct 3 sets of descriptors, that is, we need one for each
    // swapchain image.
    const auto num_frames_in_flight =
        static_cast<std::uint32_t>(_swapchain.images.size());

    // Each descriptor set has the same uniform descriptor layout.
    const auto& ubo_layout = _graphics_pipeline.desc_set_layout;
    const auto ubo_layout_handle =
        static_cast<VkDescriptorSetLayout>(ubo_layout);

    const auto desc_set_layouts = std::vector<VkDescriptorSetLayout>(
        num_frames_in_flight, ubo_layout_handle);

    _ubo_desc_sets = svk::DescriptorSets{
        desc_set_layouts.data(),  //
        num_frames_in_flight,     //
        _ubo_pool                 //
    };
  }

  auto initialize_model_view_projection_ubos() -> void
  {
    const auto num_frames_in_flight = _swapchain.images.size();
    _mvp_ubos = std::vector<svk::Buffer>(num_frames_in_flight);
    _mvp_dmems = std::vector<svk::DeviceMemory>(num_frames_in_flight);

    for (auto i = std::size_t{}; i != num_frames_in_flight; ++i)
    {
      // 1. Create UBOs.
      _mvp_ubos[i] = svk::BufferFactory{_device}
                         .make_uniform_buffer<ModelViewProjectionStack>(1);
      // 2. Allocate device memory objects for each UBO.
      _mvp_dmems[i] = svk::DeviceMemoryFactory{_physical_device, _device}
                          .allocate_for_uniform_buffer(_mvp_ubos[i]);
      // 3. Bind the buffer to the corresponding device memory objects.
      _mvp_ubos[i].bind(_mvp_dmems[i], 0);

      // 4. Get the virtual host pointer to transfer the UBO data from CPU to
      //    GPU device.
      _mvp_ubo_ptrs.emplace_back(
          _mvp_dmems[i].map_memory<ModelViewProjectionStack>(1));
    }

    for (auto i = 0u; i != num_frames_in_flight; ++i)
    {
      // 4.a) Register the byte size, the type of buffer which the descriptor
      //      references to.
      auto write_dset = VkWriteDescriptorSet{};
      write_dset.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write_dset.dstSet = _ubo_desc_sets[i];
      write_dset.dstBinding = 0;       // layout(binding = 0) uniform ...
      write_dset.dstArrayElement = 0;  // Worry about this later.
      write_dset.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      write_dset.descriptorCount = 1;  // Only 1 UBO descriptor per set.

      // 4.b) Each descriptor set being a singleton, must reference to a UBO.
      auto buffer_info = VkDescriptorBufferInfo{};
      buffer_info.buffer = _mvp_ubos[i];
      buffer_info.offset = 0;
      buffer_info.range = sizeof(ModelViewProjectionStack);
      write_dset.pBufferInfo = &buffer_info;

      // 4.c) Send this metadata to Vulkan.
      vkUpdateDescriptorSets(_device, 1, &write_dset, 0, nullptr);
    }
  }

  auto update_mvp_uniform(const std::uint32_t swapchain_image_index) -> void
  {
    static auto start_time = std::chrono::high_resolution_clock::now();

    const auto current_time = std::chrono::high_resolution_clock::now();
    const auto time =
        std::chrono::duration<float, std::chrono::seconds::period>(
            current_time - start_time)
            .count();

    auto mvp = ModelViewProjectionStack{};
    mvp.model.rotate(Eigen::AngleAxisf{time, Eigen::Vector3f::UnitZ()});

    memcpy(_mvp_ubo_ptrs[swapchain_image_index], &mvp, sizeof(mvp));
  }

  auto draw_frame() -> void
  {
    static constexpr auto forever = std::numeric_limits<std::uint64_t>::max();
    auto result = VkResult{};

    if constexpr (_verbose)
      SARA_CHECK(_current_frame);

    // The number of images in-flight is the number of swapchain images.
    // And there are as many fences as swapchain images.
    //
    // The pre-condition to start the render loop is to initialize our Vulkan
    // application as follows:
    // - every fence is reset to an unsignalled state. This is necessary to
    //   get synchronization machinery to work correctly.
    // - the current frame index is 0;

    // Wait for the GPU signal notifying that the current frame becomes
    // available.
    //
    // The function call `vkQueueSubmit(...)` at the end of this `draw_frame`
    // method uses this `_in_flight_fences[_current_frame]` fence.
    if constexpr (_verbose)
      SARA_DEBUG << "[VK] Waiting for the render fence to signal...\n";
    _render_fences[_current_frame].wait(forever);
    // This function call blocks.
    //
    // After that, the function unblocks and the program resumes its execution
    // flow on the CPU side.

    // Acquire the next image ready to be rendered.
    if constexpr (_verbose)
      SARA_DEBUG << "[VK] Acquiring the next image ready to be rendered...\n";
    auto index_of_next_image_to_render = std::uint32_t{};
    result = vkAcquireNextImageKHR(  //
        _device,                     //
        _swapchain.handle,
        forever,                                      // timeout in nanoseconds
        _image_available_semaphores[_current_frame],  // semaphore to signal
        VK_NULL_HANDLE,                               // fence to signal
        &index_of_next_image_to_render);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
      recreate_swapchain();
      return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
      throw std::runtime_error("Failed to acquire the next swapchain image!");
    if constexpr (_verbose)
      SARA_CHECK(index_of_next_image_to_render);

    update_mvp_uniform(index_of_next_image_to_render);

    // Reset the signaled fence associated to the current frame to an
    // unsignaled state. So that the GPU can reuse it to signal.
    if constexpr (_verbose)
      SARA_DEBUG << "[VK] Resetting for the render fence...\n";
    _render_fences[_current_frame].reset();

    // Reset the command buffer associated to the current frame.
    if constexpr (_verbose)
      SARA_DEBUG << "[VK] Resetting for the command buffer...\n";
    _graphics_cmd_bufs.reset(_current_frame,
                             /*VkCommandBufferResetFlagBits*/ 0);

    // Record the draw command to be performed on this swapchain image.
    if constexpr (_verbose)
      SARA_CHECK(_swapchain_fbos.fbs.size());
    const auto& descriptor_set = static_cast<VkDescriptorSet&>(
        _ubo_desc_sets[index_of_next_image_to_render]);
    record_graphics_command_buffer(
        _graphics_cmd_bufs[_current_frame],
        _swapchain_fbos[index_of_next_image_to_render], descriptor_set);

    // Submit the draw command to the graphics queue.
    if constexpr (_verbose)
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

    // 2. Ensure that drawing starts until the GPU has finished processing
    // this
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
    //      needs to reuse the same swapchain image, i.e., the one with the
    //      same index `_current_frame`,
    //
    //      the first command `vkWaitForFences(...)` at the beginning of the
    //      draw command stalls the CPU execution flow, until the current draw
    //      command submission, here, completes.
    //
    //      After which, the fence `_render_fences[_current_frame]` enters in
    //      a signaled state and un-stalls the function
    //      `vkWaitForFences(...)`.
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
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
        _framebuffer_resized)
    {
      _framebuffer_resized = false;
      recreate_swapchain();
    }
    else if (result != VK_SUCCESS)
    {
      throw std::runtime_error{fmt::format(
          "failed to present the swapchain image {}!", _current_frame)};
    }

    // Update the current frame to the next one for the next `draw_frame`
    // call.
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
                                      VkFramebuffer framebuffer,
                                      const VkDescriptorSet& descriptor_set)
      -> void
  {
    if constexpr (_verbose)
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

    if constexpr (_verbose)
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

      // Pass the VBO to the graphics pipeline.
      static const auto vbos = std::array<VkBuffer, 1>{_vbo};
      static constexpr auto offsets = std::array<VkDeviceSize, 1>{0};
      vkCmdBindVertexBuffers(command_buffer, 0,
                             static_cast<std::uint32_t>(vbos.size()),
                             vbos.data(), offsets.data());

      // Pass the EBO to the graphics pipeline.
      vkCmdBindIndexBuffer(command_buffer, _ebo, 0, VK_INDEX_TYPE_UINT16);

      // Pass the UBO to the graphics pipeline.
      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              _graphics_pipeline.pipeline_layout,  //
                              0, 1,             // Find out later about this.
                              &descriptor_set,  //
                              0, nullptr);      // Find out later about this.

      // Tell the graphics pipeline to draw triangles.
      vkCmdDrawIndexed(command_buffer,
                       static_cast<std::uint32_t>(indices.size()), 1, 0, 0, 0);
    }

    if constexpr (_verbose)
      SARA_DEBUG << "[VK] End render pass...\n";
    vkCmdEndRenderPass(command_buffer);

    status = vkEndCommandBuffer(command_buffer);
    if (status != VK_SUCCESS)
      throw std::runtime_error{fmt::format(
          "[VK] Error: failed to end record command buffer! Error code: {}",
          static_cast<int>(status))};
  }

private: /* Swapchain recreation */
  static auto framebuffer_resize_callback(GLFWwindow* window,
                                          [[maybe_unused]] const int width,
                                          [[maybe_unused]] const int height)
      -> void
  {
    auto app = reinterpret_cast<VulkanTriangleRenderer*>(
        glfwGetWindowUserPointer(window));
    app->_framebuffer_resized = true;
  }

  auto recreate_swapchain() -> void
  {
    auto w = int{};
    auto h = int{};
    while (w == 0 || h == 0)
    {
      glfwGetFramebufferSize(_window, &w, &h);
      glfwWaitEvents();
    }
    vkDeviceWaitIdle(_device);

    // It is not possible to create two swapchains apparently, so we have to
    // destroy the current swapchain.
    if constexpr (_verbose)
      SARA_DEBUG << "DESTROYING THE CURRENT SWAPCHAIN...\n";
    _swapchain_fbos.destroy();
    _swapchain.destroy();

    if constexpr (_verbose)
      SARA_DEBUG << "RECREATING THE SWAPCHAIN (with the correct image "
                    "dimensions)...\n";
    init_swapchain(_window);
    init_swapchain_fbos();
  }

private:
  GLFWwindow* _window = nullptr;
  int _current_frame = 0;
  bool _framebuffer_resized = false;
  static constexpr bool _verbose = false;

  // Geometry data (quad)
  svk::Buffer _vbo;
  svk::Buffer _ebo;

  svk::DeviceMemory _vdm;
  svk::DeviceMemory _edm;

  // Model-view-projection matrix
  //
  // 1. UBO and device memory objects.
  std::vector<svk::Buffer> _mvp_ubos;
  std::vector<svk::DeviceMemory> _mvp_dmems;
  std::vector<void*> _mvp_ubo_ptrs;
  // 2. Layout binding referenced for the shader.
  svk::DescriptorPool _ubo_pool;
  svk::DescriptorSets _ubo_desc_sets;
};


auto main(int, char** argv) -> int
{
  SignalHandler::init();

  try
  {
    const auto app = glfw::Application{};
    app.init_for_vulkan_rendering();

    const auto app_name = "Vulkan Triangle";
    auto window = glfw::Window{300, 300, app_name};

    const auto program_dir_path = fs::absolute(fs::path(argv[0])).parent_path();
    auto triangle_renderer = VulkanTriangleRenderer{
        window, app_name, program_dir_path / "hello_vulkan_triangle_shaders",
        true};
    triangle_renderer.loop(window);
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
