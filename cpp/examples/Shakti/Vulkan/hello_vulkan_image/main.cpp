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
#include <DO/Shakti/Vulkan/DeviceMemory.hpp>
#include <DO/Shakti/Vulkan/EasyGLFW.hpp>
#include <DO/Shakti/Vulkan/GraphicsBackend.hpp>
#include <DO/Shakti/Vulkan/Image.hpp>
#include <DO/Shakti/Vulkan/ImageView.hpp>
#include <DO/Shakti/Vulkan/Sampler.hpp>

#include <DO/Kalpana/Math/Projection.hpp>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/VideoIO.hpp>


namespace k = DO::Kalpana;
namespace sara = DO::Sara;
namespace glfw = DO::Kalpana::GLFW;
namespace kvk = DO::Kalpana::Vulkan;
namespace svk = DO::Shakti::Vulkan;
namespace fs = std::filesystem;


//! @brief The 4 vertices of the square.
// clang-format off
static auto vertices = std::vector<Vertex>{
  {.pos = {-0.5f, -0.5f}, .uv = {0.f, 0.f}},
  {.pos = { 0.5f, -0.5f}, .uv = {1.f, 0.f}},
  {.pos = { 0.5f,  0.5f}, .uv = {1.f, 1.f}},
  {.pos = {-0.5f,  0.5f}, .uv = {0.f, 1.f}}
};
// clang-format on

//! @brief The 2 triangles to form the square.
static const auto indices = std::vector<uint16_t>{
    0, 1, 2,  //
    2, 3, 0   //
};


class VulkanImagePipelineBuilder : public kvk::GraphicsPipeline::Builder
{
  using BaseBuilder = kvk::GraphicsPipeline::Builder;

public:
  VulkanImagePipelineBuilder(const svk::Device& device,
                             const kvk::RenderPass& render_pass)
    : BaseBuilder{device, render_pass}
  {
  }

  //! @brief Focus the attention here.
  auto create() -> kvk::GraphicsPipeline override
  {
    load_shaders();
    initialize_fixed_functions();

    auto graphics_pipeline = kvk::GraphicsPipeline{};

    graphics_pipeline.device = device;

    graphics_pipeline.desc_set_layout =
        svk::DescriptorSetLayout::Builder{device}
            .push_uniform_buffer_layout_binding(0)
            .push_image_sampler_layout_binding(1)
            .create();

    create_graphics_pipeline_layout(graphics_pipeline);
    create_graphics_pipeline(graphics_pipeline);

    return graphics_pipeline;
  }
};


class VulkanImageRenderer : public kvk::GraphicsBackend
{
public:
  VulkanImageRenderer(GLFWwindow* window, const std::string& app_name,
                      const std::filesystem::path& program_dir_path,
                      const std::filesystem::path& video_path,
                      const bool debug_vulkan = true)
    : _window{window}
    , _verbose{debug_vulkan}
    , _vpath{video_path}
  {
    // Set the GLFW callbacks.
    {
      glfwSetWindowUserPointer(window, this);
      glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);
    }

    _vstream.open(_vpath);
    const auto image_host = sara::from_rgb8_to_rgba8(_vstream.frame());

    const auto aspect_ratio =
        static_cast<float>(image_host.width()) / image_host.height();
    for (auto& vertex : vertices)
      vertex.pos.x() *= aspect_ratio;

    // General vulkan context objects.
    init_instance(app_name, debug_vulkan);
    init_surface(window);
    init_physical_device();
    init_device_and_queues();
    init_swapchain(window);
    init_render_pass();
    init_swapchain_fbos();

    // Graphics pipeline.
    const auto shader_dir_path =
        program_dir_path / "hello_vulkan_image_shaders";
    init_graphics_pipeline(window,  //
                           shader_dir_path / "vert.spv",
                           shader_dir_path / "frag.spv");

    // Graphics command pool and command buffers.
    init_command_pool_and_buffers();
    // Vulkan fence and semaphores for graphics rendering.
    init_synchronization_objects();

    // Geometry data.
    init_vbos(vertices);
    init_ebos(indices);

    // Device memory for image data..
    init_vulkan_image_objects(image_host);
    init_image_copy_command_buffers();
    // Initialize the image data on the device side.
    copy_image_data_from_host_to_staging_buffer(image_host);
    copy_image_data_from_staging_to_device_buffer();

    // Graphics pipeline resource objects.
    make_descriptor_pool();
    make_descriptor_sets();

    // Uniform data for the shaders.
    //
    // 1. Model-view-projection matrices
    init_mvp_ubos();
    // 2. Image sampler objects
    init_image_view_and_sampler();

    define_descriptor_set_types();
  }

  auto loop(GLFWwindow* window) -> void
  {
    while (!glfwWindowShouldClose(window))
    {
      glfwPollEvents();

      if (_vstream.read())
      {
        // if (_verbose)
        sara::tic();
        const auto image_host = sara::from_rgb8_to_rgba8(_vstream.frame());
        // if (_verbose)
        sara::toc("RGB to RGBA");

        if (_verbose)
          sara::tic();
        copy_image_data_from_host_to_staging_buffer(image_host);
        copy_image_data_from_staging_to_device_buffer();
        if (_verbose)
          sara::toc("Transfer from host to device image");
      }

      draw_frame();

      if (SignalHandler::ctrl_c_hit)
        break;
    }

    vkDeviceWaitIdle(_device);
  }

private: /* Methods to initialize objects for the graphics pipeline. */
  auto init_device_and_queues() -> void override
  {
    // According to:
    // https://stackoverflow.com/questions/61434615/in-vulkan-is-it-beneficial-for-the-graphics-queue-family-to-be-separate-from-th
    //
    // Using distinct queue families, namely one for the graphics operations
    // and another for the present operations, does not result in better
    // performance.
    //
    // This is because the hardware does not expose present-only queue
    // families...
    const auto graphics_queue_family_index =
        kvk::find_graphics_queue_family_indices(_physical_device).front();
    const auto present_queue_family_index =
        kvk::find_present_queue_family_indices(_physical_device, _surface)
            .front();
    const auto queue_family_indices = std::set{
        graphics_queue_family_index,  //
        present_queue_family_index    //
    };

    auto device_extensions = std::vector{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (compile_for_apple)
      device_extensions.emplace_back("VK_KHR_portability_subset");

    // Here we also require the anisotropic sampling capability from the
    // hardware.
    auto device_features = VkPhysicalDeviceFeatures{};
    device_features.samplerAnisotropy = VK_TRUE;

    // Create a logical device with our requirements.
    _device = svk::Device::Builder{_physical_device}
                  .enable_device_extensions(device_extensions)
                  .enable_queue_families(queue_family_indices)
                  .enable_physical_device_features(device_features)
                  .enable_validation_layers(_validation_layers)
                  .create();

    SARA_DEBUG << "[VK] - Initializing the graphics queue...\n";
    _graphics_queue = svk::Queue{_device, graphics_queue_family_index};
    SARA_DEBUG << "[VK] - Initializing the present queue...\n";
    _present_queue = svk::Queue{_device, present_queue_family_index};
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
        VulkanImagePipelineBuilder{_device, _render_pass}
            .vertex_shader_path(vertex_shader_path)
            .fragment_shader_path(fragment_shader_path)
            .vbo_data_format<Vertex>()
            .input_assembly_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
            .viewport_sizes(static_cast<float>(w), static_cast<float>(h))
            .scissor_sizes(w, h)
            .create();
  }

  auto make_descriptor_pool() -> void
  {
    const auto num_frames_in_flight =
        static_cast<std::uint32_t>(_swapchain.images.size());  // 3 typically.

    // We just need a single descriptor pool.
    auto desc_pool_builder = svk::DescriptorPool::Builder{_device}  //
                                 .pool_count(2)
                                 .pool_max_sets(num_frames_in_flight);
    // This descriptor pool can only allocate UBO descriptors.
    desc_pool_builder.pool_type(0) = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // The cumulated number of UBO descriptors across all every descriptor sets
    // cannot exceed the following number of descriptors (3).
    desc_pool_builder.descriptor_count(0) = num_frames_in_flight;

    // The second descriptor pool is dedicated to the allocation of Vulkan
    // image descriptors.
    desc_pool_builder.pool_type(1) = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    // We only need to reserve 1 image descriptor since the image will stay
    // unchanged throughout the application runtime
    desc_pool_builder.descriptor_count(1) = num_frames_in_flight;

    _desc_pool = desc_pool_builder.create();
  }

  auto make_descriptor_sets() -> void
  {
    // The number of frames in flight is the number of swapchain images.
    // Let's say there are 3 frames in flight.
    //
    // We will construct 3 sets of descriptors, that is, we need one for each
    // swapchain image.
    const auto num_frames_in_flight =
        static_cast<std::uint32_t>(_swapchain.images.size());

    // Each descriptor set has the same uniform descriptor layout.
    const auto& dset_layout = _graphics_pipeline.desc_set_layout;
    const auto dset_layout_handle =
        static_cast<VkDescriptorSetLayout>(dset_layout);

    const auto dset_layouts = std::vector<VkDescriptorSetLayout>(
        num_frames_in_flight, dset_layout_handle);

    _desc_sets = svk::DescriptorSets{
        dset_layouts.data(),   //
        num_frames_in_flight,  //
        _desc_pool             //
    };
  }

  auto define_descriptor_set_types() -> void
  {
    const auto num_frames_in_flight = _swapchain.images.size();
    for (auto i = std::size_t{}; i != num_frames_in_flight; ++i)
    {
      // 1. Descriptor set #1: the model-view-projection matrix stack uniform.
      auto buffer_info = VkDescriptorBufferInfo{};
      buffer_info.buffer = _mvp_ubos[i];
      buffer_info.offset = 0;
      buffer_info.range = sizeof(ModelViewProjectionStack);

      // 2. Descriptor set #2: the image sampler uniform.
      auto image_info = VkDescriptorImageInfo{};
      image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      image_info.imageView = _image_view;
      image_info.sampler = _image_sampler;

      auto write_dsets = std::array<VkWriteDescriptorSet, 2>{};

      // 3. Register the byte size, the type of buffer which the descriptor
      //    references to.
      auto& mvp_wdset = write_dsets[0];
      mvp_wdset.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      mvp_wdset.dstSet = _desc_sets[i];
      mvp_wdset.dstBinding = 0;       // layout(binding = 0) uniform ...
      mvp_wdset.dstArrayElement = 0;  // Worry about this later.
      mvp_wdset.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      mvp_wdset.descriptorCount = 1;  // Only 1 UBO descriptor per set.
      mvp_wdset.pBufferInfo = &buffer_info;

      auto& image_wdset = write_dsets[1];
      image_wdset.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      image_wdset.dstSet = _desc_sets[i];
      image_wdset.dstBinding = 1;
      image_wdset.dstArrayElement = 0;
      image_wdset.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      image_wdset.descriptorCount = 1;
      image_wdset.pImageInfo = &image_info;

      // 4. Send this metadata to Vulkan.
      vkUpdateDescriptorSets(_device,
                             static_cast<std::uint32_t>(write_dsets.size()),
                             write_dsets.data(), 0, nullptr);
    }
  }

private: /* Methods to initialize geometry buffer data on the device side. */
  auto init_vbos(const std::vector<Vertex>& vertices) -> void
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

  auto init_ebos(const std::vector<std::uint16_t>& indices) -> void
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

private: /* Methods to transfer model-view-projection uniform data. */
  auto init_mvp_ubos() -> void
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
  }

  //! @brief Every time we render the image frame through the method
  //! `VulkanImageRenderer::draw_frame`, we update the model-view-projection
  //! matrix stack by calling this method and therefore animate the square.
  auto update_mvp_ubo(const std::uint32_t swapchain_image_index) -> void
  {
    // static auto start_time = std::chrono::high_resolution_clock::now();

    // const auto current_time = std::chrono::high_resolution_clock::now();
    // const auto time =
    //     std::chrono::duration<float, std::chrono::seconds::period>(
    //         current_time - start_time)
    //         .count();

    // _mvp.model.rotate(Eigen::AngleAxisf{time, Eigen::Vector3f::UnitZ()});

    memcpy(_mvp_ubo_ptrs[swapchain_image_index], &_mvp, sizeof(_mvp));
  }

private: /* Methods to initialize image data */
  auto init_vulkan_image_objects(const sara::ImageView<sara::Rgba8>& image_host)
      -> void
  {
    // Temporary image object on the host side.
    const auto w = static_cast<std::uint32_t>(image_host.width());
    const auto h = static_cast<std::uint32_t>(image_host.height());

    // The image object as a staging device buffer.
    _image_staging_buffer =
        svk::BufferFactory{_device}  //
            .make_staging_buffer<std::uint32_t>(image_host.size());
    _image_staging_dmem =
        svk::DeviceMemoryFactory{_physical_device, _device}  //
            .allocate_for_staging_buffer(_image_staging_buffer);
    _image_staging_buffer.bind(_image_staging_dmem, 0);

    // Copy the image data from the host buffer to the staging device buffer.
    _image_staging_dmem.copy_from(image_host.data(), image_host.size());

    // Image data as device image associated with a device memory.
    _image =
        svk::Image::Builder{_device}
            .sizes(VkExtent2D{w, h})
            .format(VK_FORMAT_R8G8B8A8_SRGB)
            .tiling(VK_IMAGE_TILING_OPTIMAL)
            .usage(VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)
            .create();
    _image_dmem = svk::DeviceMemoryFactory{_physical_device, _device}  //
                      .allocate_for_device_image(_image);
    _image.bind(_image_dmem, 0);
  }

  auto init_image_view_and_sampler() -> void
  {
    // To use the image resource from a shader:
    // 1. Create an image view
    // 2. Create an image sampler
    // 3. Add a DescriptorSetLayout for the image sampler.
    _image_view = svk::ImageView::Builder{_device}
                      .image(_image)
                      .format(VK_FORMAT_R8G8B8A8_SRGB)
                      .aspect_mask(VK_IMAGE_ASPECT_COLOR_BIT)
                      .create();

    _image_sampler = svk::Sampler::Builder{_physical_device, _device}.create();
  }

  auto init_image_copy_command_buffers() -> void
  {
    _image_copy_cmd_bufs = svk::CommandBufferSequence{
        1,                  //
        _device,            //
        _graphics_cmd_pool  //
    };
  }

  auto copy_image_data_from_host_to_staging_buffer(
      const sara::ImageView<sara::Rgba8>& image_host) -> void
  {
    // Temporary image object on the host side.
    _image_staging_dmem.copy_from(image_host.data(), image_host.size(), 0);
  }

  auto copy_image_data_from_staging_to_device_buffer() -> void
  {
    _image_copy_cmd_bufs.reset(0);
    {
      auto& image_layout_transition_cmd_buf = _image_copy_cmd_bufs[0];
      svk::record_image_layout_transition(_image,  //
                                          VK_IMAGE_LAYOUT_UNDEFINED,
                                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                          image_layout_transition_cmd_buf);
      _graphics_queue.submit_commands(_image_copy_cmd_bufs);
      _graphics_queue.wait();
    }

    _image_copy_cmd_bufs.reset(0);
    {
      auto& image_copy_cmd_buf = _image_copy_cmd_bufs[0];
      svk::record_copy_buffer_to_image(_image_staging_buffer, _image,
                                       image_copy_cmd_buf);
      _graphics_queue.submit_commands(_image_copy_cmd_bufs);
      _graphics_queue.wait();
    }

    _image_copy_cmd_bufs.reset(0);
    {
      auto& image_layout_transition_cmd_buf = _image_copy_cmd_bufs[0];
      svk::record_image_layout_transition(
          _image,  //
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,  //
          image_layout_transition_cmd_buf);
      _graphics_queue.submit_commands(_image_copy_cmd_bufs);
      _graphics_queue.wait();
    }
  }

private: /* Methods for onscreen rendering */
  auto record_graphics_command_buffer(VkCommandBuffer command_buffer,
                                      VkFramebuffer framebuffer,
                                      const VkDescriptorSet& descriptor_set)
      -> void
  {
    if (_verbose)
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

    if (_verbose)
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

    if (_verbose)
      SARA_DEBUG << "[VK] End render pass...\n";
    vkCmdEndRenderPass(command_buffer);

    status = vkEndCommandBuffer(command_buffer);
    if (status != VK_SUCCESS)
      throw std::runtime_error{fmt::format(
          "[VK] Error: failed to end record command buffer! Error code: {}",
          static_cast<int>(status))};
  }

  auto draw_frame() -> void
  {
    static constexpr auto forever = std::numeric_limits<std::uint64_t>::max();
    auto result = VkResult{};

    if (_verbose)
      SARA_CHECK(_current_frame);

    // The number of images in-flight is the number of swapchain images.
    // And there are as many fences as swapchain images.
    //
    // The pre-condition to start the render loop is to initialize our Vulkan
    // application as follows:
    // - every fence is reset to an unsignalled state. This is necessary to
    //   get synchronization machinery to work correctly.
    // - the current frame index is 0;

    // Wait for the GPU signal that the current frame becomes available.
    //
    // The function call `vkQueueSubmit(...)` at the end of this `draw_frame`
    // method uses this `_in_flight_fences[_current_frame]` fence.
    if (_verbose)
      SARA_DEBUG << "[VK] Waiting for the render fence to signal...\n";
    _render_fences[_current_frame].wait(forever);
    // This function call blocks.
    //
    // After that, the function unblocks and the program resumes its execution
    // flow on the CPU side.

    // Acquire the next image ready to be rendered.
    if (_verbose)
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
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
      recreate_swapchain();
      return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
      throw std::runtime_error("Failed to acquire the next swapchain image!");
    if (_verbose)
      SARA_CHECK(index_of_next_image_to_render);

    update_mvp_ubo(index_of_next_image_to_render);

    // Reset the signaled fence associated to the current frame to an
    // unsignaled state. So that the GPU can reuse it to signal.
    if (_verbose)
      SARA_DEBUG << "[VK] Resetting for the render fence...\n";
    _render_fences[_current_frame].reset();

    // Reset the command buffer associated to the current frame.
    if (_verbose)
      SARA_DEBUG << "[VK] Resetting for the command buffer...\n";
    _graphics_cmd_bufs.reset(_current_frame,
                             /*VkCommandBufferResetFlagBits*/ 0);

    // Record the draw command to be performed on this swapchain image.
    if (_verbose)
      SARA_CHECK(_swapchain_fbos.fbs.size());
    const auto& descriptor_set = static_cast<VkDescriptorSet&>(
        _desc_sets[index_of_next_image_to_render]);
    record_graphics_command_buffer(
        _graphics_cmd_bufs[_current_frame],
        _swapchain_fbos[index_of_next_image_to_render], descriptor_set);

    // Submit the draw command to the graphics queue.
    if (_verbose)
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

private: /* Swapchain recreation */
  static auto framebuffer_resize_callback(GLFWwindow* window,
                                          [[maybe_unused]] const int width,
                                          [[maybe_unused]] const int height)
      -> void
  {
    auto app = reinterpret_cast<VulkanImageRenderer*>(
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
    if (_verbose)
      SARA_DEBUG << "DESTROYING THE CURRENT SWAPCHAIN...\n";
    _swapchain_fbos.destroy();
    _swapchain.destroy();

    if (_verbose)
      SARA_DEBUG << "RECREATING THE SWAPCHAIN (with the correct image "
                    "dimensions)...\n";
    init_swapchain(_window);
    init_swapchain_fbos();

    const auto fb_aspect_ratio = static_cast<float>(w) / h;
    _mvp.projection = k::orthographic(                    //
        -0.5f * fb_aspect_ratio, 0.5f * fb_aspect_ratio,  //
        -0.5f, 0.5f,                                      //
        -0.5f, 0.5f);

    SARA_CHECK(_mvp.model.matrix());
    SARA_CHECK(_mvp.view.matrix());
    SARA_CHECK(_mvp.projection);
  }

private:
  GLFWwindow* _window = nullptr;
  int _current_frame = 0;
  bool _framebuffer_resized = false;
  bool _verbose = false;

  std::filesystem::path _vpath;
  sara::VideoStream _vstream;

  // Geometry data (quad)
  svk::Buffer _vbo;
  svk::DeviceMemory _vdm;

  svk::Buffer _ebo;
  svk::DeviceMemory _edm;

  // Model-view-projection matrix
  //
  // 1. UBO and device memory objects.
  ModelViewProjectionStack _mvp;
  std::vector<svk::Buffer> _mvp_ubos;
  std::vector<svk::DeviceMemory> _mvp_dmems;
  std::vector<void*> _mvp_ubo_ptrs;

  // 2.a) Staging image buffer and its allocated memory.
  svk::Buffer _image_staging_buffer;
  svk::DeviceMemory _image_staging_dmem;

  // 2.b) Device image and its allocated memory
  svk::Image _image;
  svk::DeviceMemory _image_dmem;
  // 2.c) Device image bindings for the shader.
  svk::ImageView _image_view;
  svk::Sampler _image_sampler;

  // 2.d) Copy command buffers.
  svk::CommandBufferSequence _image_copy_cmd_bufs;

  // 3. Layout binding referenced for the shader.
  svk::DescriptorPool _desc_pool;
  svk::DescriptorSets _desc_sets;
};


auto main(int argc, char** argv) -> int
{
  SignalHandler::init();

  if (argc < 2)
  {
    std::cerr << "USAGE: " << argv[0] << " VIDEO_PATH\n";
    return 0;
  }

  try
  {
    const auto app = glfw::Application{};
    app.init_for_vulkan_rendering();

    const auto app_name = "Vulkan Image";
    auto window = glfw::Window{800, 600, app_name};

    const auto program_dir_path = fs::absolute(fs::path(argv[0])).parent_path();
    const auto video_path = fs::path(argv[1]);
    static constexpr auto debug = true;
    auto triangle_renderer = VulkanImageRenderer{
        window,            //
        app_name,          //
        program_dir_path,  //
        video_path,        //
        debug              //
    };
    triangle_renderer.loop(window);
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
