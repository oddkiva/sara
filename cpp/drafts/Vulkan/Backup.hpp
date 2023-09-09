VkCommandPool command_pool;

VkBuffer vertex_buffer;
VkDeviceMemory vertex_buffer_memory;

std::vector<VkCommandBuffer> command_buffers;

std::vector<VkSemaphore> image_available_semaphores;
std::vector<VkSemaphore> render_finished_semaphores;
std::vector<VkFence> in_flight_fences;
std::vector<VkFence> images_in_flight;
size_t current_frame = 0;

void init_vulkan()
{
  create_swapchain();
  create_image_views();
  create_render_pass();
  create_graphics_pipeline();
  create_framebuffers();
  create_command_pool();
  create_vertex_buffer();
  create_command_buffers();
  create_sync_objects();
}

void mainLoop()
{
  while (!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    draw_frame();
  }

  vkDeviceWaitIdle(_device);
}

void cleanup()
{
  cleanup_swap_chain();

  vkDestroyBuffer(_device, vertexBuffer, nullptr);
  vkFreeMemory(_device, vertexBufferMemory, nullptr);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
  {
    vkDestroySemaphore(_device, renderFinishedSemaphores[i], nullptr);
    vkDestroySemaphore(_device, imageAvailableSemaphores[i], nullptr);
    vkDestroyFence(_device, inFlightFences[i], nullptr);
  }

  vkDestroyCommandPool(_device, commandPool, nullptr);

  vkDestroyDevice(_device, nullptr);

  if (enable_validation_layers)
  {
    DestroyDebugUtilsMessengerEXT(_instance, _debug_messenger, nullptr);
  }

  vkDestroySurfaceKHR(_instance, _surface, nullptr);
  vkDestroyInstance(_instance, nullptr);

  glfwDestroyWindow(window);

  glfwTerminate();
}

void recreate_swapchain()
{
  int width = 0, height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  while (width == 0 || height == 0)
  {
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
  }

  vkDeviceWaitIdle(_device);

  cleanup_swap_chain();

  createSwapChain();
  createImageViews();
  createRenderPass();
  createGraphicsPipeline();
  createFramebuffers();
  createCommandBuffers();

  imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
}


// Command pool for graphics family queue.
auto create_command_pool() -> void
{
  const auto queue_family_indices = find_queue_families(_physical_device);

  auto pool_info = VkCommandPoolCreateInfo{};
  {
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family_indices.graphicsFamily.value();
    pool_info.flags = 0;
  }

  if (vkCreateCommandPool(_device, &pool_info, nullptr, &_command_pool) !=
      VK_SUCCESS)
    throw std::runtime_error{"Failed to create command pool!"};
}

auto create_command_buffers() -> void
{
  _command_buffers.resize(_swapchain_framebuffers.size());

  auto alloc_info = VkCommandBufferAllocateInfo{};
  {
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = _command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount =
        static_cast<std::uint32_t>(_command_buffers.size());
  }

  if (vkAllocateCommandBuffers(_device, &alloc_info, _command_buffers.data()) !=
      VK_SUCCESS)
    throw std::runtime_error{"Failed to allocate command buffers!"};

  for (auto i = 0u; i < _command_buffers.size(); ++i)
  {
    auto begin_info = VkCommandBufferBeginInfo{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = 0;
    begin_info.pInheritanceInfo = nullptr;

    if (vkBeginCommandBuffer(_command_buffers[i], &begin_info) != VK_SUCCESS)
    {
      throw std::runtime_error
      {
        "Failed to begin recording command
            buffer !"};

            auto render_pass_begin_info = VkRenderPassBeginInfo{};
        {
          render_pass_begin_info.sType =
              VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
          render_pass_begin_info.renderPass = _render_pass;
          render_pass_begin_info.framebuffer = _swapchain_framebuffers[i];
          render_pass_begin_info.renderArea.offset = {0, 0};
          render_pass_begin_info.renderArea.extent = _swapchain_extent;

          auto clear_color = VkClearValue{{{0.f, 0.f, 0.f, 1.f}}};
          render_pass_begin_info.clearValueCount = 1;
          render_pass_begin_info.pClearValues = &clear_color;
        }

        vkCmdBeginRenderPass(_command_buffers[i], &render_pass_begin_info,
                             VK_SUBPASS_CONTENTS_INLINE);
        {
          vkCmdBindPipeline(_command_buffers[i],
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            _graphics_pipeline);
          vkCmdDraw(_command_buffers[i], 3, 1, 0, 0);
        }
        vkCmdEndRenderPass(_command_buffers[i]);

        if (vkEndCommandBuffer(_command_buffers[i]) != VK_SUCCESS)
          throw std::runtime_error{"Failed to record command buffer!"};
      }
    }
  }
}

// Semaphores
auto create_sync_objects() -> void
{
  _image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
  _render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
  _in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
  _images_in_flight.resize(_swapchain_images.size(), VK_NULL_HANDLE);

  auto semaphore_info = VkSemaphoreCreateInfo{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  auto fence_info = VkFenceCreateInfo{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (auto i = 0u; i < MAX_FRAMES_IN_FLIGHT; ++i)
  {
    if (vkCreateSemaphore(_device, &semaphore_info, nullptr,
                          &_image_available_semaphores[i]) != VK_SUCCESS ||
        vkCreateSemaphore(_device, &semaphore_info, nullptr,
                          &_render_finished_semaphores[i]) != VK_SUCCESS ||
        vkCreateFence(_device, &fence_info, nullptr, &_in_flight_fences[i]) !=
            VK_SUCCESS)
    {
      throw std::runtime_error{
          "Failed to create synchronization objects for a frame!"};
    }
  }
}


//
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
  auto result = vkAcquireNextImageKHR(
      _device, _swapchain, UINT64_MAX,
      _image_available_semaphores[current_frame], VK_NULL_HANDLE, &image_index);
  if (result == VK_ERROR_OUT_OF_DATE_KHR)
  {
    // For example when the window is resized, we need to recreate the
    swap
        // chain, the images of the swap chains must reinitialized with the
        new
        // window extent.
        recreate_swapchain();
    return;
  }
  else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    throw std::runtime_error("failed to acquire swap chain image!");

  // The CPU encounters another fence controlled by the GPU:
  // - The CPU is waiting from the GPU until the image becomes available
    for
      //   rendering.
      // - The second fence displays a red light
      // - When rendering for the first time, the second fence is already
      opened,
          //   so the CPU can pursue its journey.
          if (_images_in_flight[image_index] != VK_NULL_HANDLE)
              vkWaitForFences(_device, 1, &_images_in_flight[current_frame],
                              VK_TRUE, UINT64_MAX);
    _images_in_flight[image_index] = _in_flight_fences[current_frame];
    // - The second fence has opened just now.

    // The CPU pursues its journey:
    //
    // - The CPU will now detail what to do at the drawing stage.
    auto submit_info = VkSubmitInfo{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkPipelineStageFlags wait_stages =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // - The CPU tells the GPU to ensure that it starts drawing only when
    the
        // image becomes available.
        submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &_image_available_semaphores[current_frame];
    submit_info.pWaitDstStageMask = &wait_stages;

    // - The CPU tells the GPU to ensure that it notifies when the drawing
    is
        //   finished and thus ready to present onto the screen.
        submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &_render_finished_semaphores[current_frame];

    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &_command_buffers[image_index];

    // - The CPU now tells the GPU to make the first fence close the road.
    vkResetFences(_device, 1, &_in_flight_fences[current_frame]);

    // - The CPU submits a drawing command to the GPU (on the graphics
    queue). if (vkQueueSubmit(_graphics_queue, 1, &submit_info,
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


//
auto recreate_swapchain() -> void
{
  vkDeviceWaitIdle(_device);

  cleanup_swapchain();

  create_swapchain();
  create_image_views();
  create_render_pass();
  create_graphics_pipeline();
  create_framebuffers();
  create_command_buffers();
}

auto cleanup_swapchain() -> void
{
  for (auto& framebuffer : _swapchain_framebuffers)
    vkDestroyFramebuffer(_device, framebuffer, nullptr);

  vkFreeCommandBuffers(_device, _command_pool,
                       static_cast<std::uint32_t>(_command_buffers.size()),
                       _command_buffers.data());

  vkDestroyPipeline(_device, _graphics_pipeline, nullptr);
  vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
  vkDestroyRenderPass(_device, _render_pass, nullptr);

  for (auto& image_view : _swapchain_image_views)
    vkDestroyImageView(_device, image_view, nullptr);

  vkDestroySwapchainKHR(_device, _swapchain, nullptr);
}

VkRenderPass _render_pass;
VkPipelineLayout _pipeline_layout;
VkPipeline _graphics_pipeline;

VkCommandPool _command_pool;
std::vector<VkCommandBuffer> _command_buffers;

//! @brief "Traffic lights".
//! @{
std::vector<VkSemaphore> _image_available_semaphores;
std::vector<VkSemaphore> _render_finished_semaphores;
std::vector<VkFence> _in_flight_fences;
std::vector<VkFence> _images_in_flight;
//! @}

std::int32_t current_frame = 0;
