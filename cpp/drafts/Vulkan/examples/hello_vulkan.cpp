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
  void run()
  {
    init();
    cleanup();
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
