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

#define BOOST_TEST_MODULE "Vulkan/Graphics Backend"
#define GLFW_INCLUDE_VULKAN

#include <DO/Shakti/Vulkan/EasyGLFW.hpp>
#include <DO/Shakti/Vulkan/GraphicsBackend.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>

#include <boost/test/unit_test.hpp>


namespace kvk = DO::Kalpana::Vulkan;


struct Vertex
{
  Eigen::Vector2f pos;
  Eigen::Vector3f color;

  static auto get_binding_description() -> VkVertexInputBindingDescription
  {
    VkVertexInputBindingDescription binding_description{};
    binding_description.binding = 0;
    binding_description.stride = sizeof(Vertex);
    binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return binding_description;
  }

  static auto get_attribute_descriptions()
      -> std::vector<VkVertexInputAttributeDescription>
  {
    auto attribute_descriptions =
        std::vector<VkVertexInputAttributeDescription>(2);

    // Position
    attribute_descriptions[0].binding = 0;
    attribute_descriptions[0].location = 0;
    attribute_descriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attribute_descriptions[0].offset = offsetof(Vertex, pos);

    // Color
    attribute_descriptions[1].binding = 0;
    attribute_descriptions[1].location = 1;
    attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute_descriptions[1].offset = offsetof(Vertex, color);

    return attribute_descriptions;
  }
};

class MyGraphicsBackend : public kvk::GraphicsBackend
{
public:
  MyGraphicsBackend(GLFWwindow* window, const std::string& app_name,
                    const std::filesystem::path& vertex_shader_path,
                    const std::filesystem::path& fragment_shader_path,
                    const bool debug_vulkan)
  {
    init_instance(app_name, debug_vulkan);
    init_surface(window);
    init_physical_device();
    init_device_and_queues();
    init_swapchain(window);
    init_render_pass();
    init_framebuffers();
    init_graphics_pipeline(window, vertex_shader_path, fragment_shader_path);
    init_command_pool_and_buffers();
    init_synchronization_objects();
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
            // .vbo_data_built_in_vertex_shader()
            .vbo_data_format<Vertex>()
            .input_assembly_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
            .viewport_sizes(static_cast<float>(w), static_cast<float>(h))
            .scissor_sizes(w, h)
            .create();
  }
};


BOOST_AUTO_TEST_CASE(test_graphics_backend)
{
  namespace glfw = DO::Kalpana::GLFW;

  auto glfw_app = glfw::Application{};
  glfw_app.init_for_vulkan_rendering();

  // Create a window.
  const auto window = glfw::Window(100, 100, "Vulkan");

  namespace fs = std::filesystem;
  static const auto program_path =
      fs::path(boost::unit_test::framework::master_test_suite().argv[0]);
  static const auto shader_dir_path =
      program_path.parent_path() / "test_shaders";
  static const auto vshader_path = shader_dir_path / "vert.spv";
  static const auto fshader_path = shader_dir_path / "frag.spv";

  static constexpr auto with_vulkan_logging = true;

  const auto vk_backend = MyGraphicsBackend{
      window,
      "GLFW-Vulkan App",  //
      vshader_path, fshader_path,
      with_vulkan_logging  //
  };
}
