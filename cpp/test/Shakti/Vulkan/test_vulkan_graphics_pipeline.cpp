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

#define BOOST_TEST_MODULE "Vulkan/Graphics Pipeline"
#define GLFW_INCLUDE_VULKAN

#include <DO/Shakti/Vulkan/Device.hpp>
#include <DO/Shakti/Vulkan/EasyGLFW.hpp>
#include <DO/Shakti/Vulkan/GraphicsPipeline.hpp>
#include <DO/Shakti/Vulkan/Instance.hpp>
#include <DO/Shakti/Vulkan/PhysicalDevice.hpp>
#include <DO/Shakti/Vulkan/RenderPass.hpp>
#include <DO/Shakti/Vulkan/Surface.hpp>
#include <DO/Shakti/Vulkan/Swapchain.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>

#include <boost/test/unit_test.hpp>


static constexpr auto debug_vulkan_instance = true;
#if defined(__APPLE__)
static constexpr auto compile_for_apple = true;
#else
static constexpr auto compile_for_apple = false;
#endif


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


BOOST_AUTO_TEST_CASE(test_graphics_pipeline_build)
{
  namespace svk = DO::Shakti::Vulkan;
  namespace k = DO::Kalpana;
  namespace glfw = k::GLFW;
  namespace kvk = DO::Kalpana::Vulkan;

  auto glfw_app = glfw::Application{};
  glfw_app.init_for_vulkan_rendering();

  // Create a window.
  const auto window = glfw::Window(100, 100, "Vulkan");

  // Vulkan instance.
  auto instance_extensions =
      kvk::Surface::list_required_instance_extensions_from_glfw();
  if constexpr (debug_vulkan_instance)
    instance_extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  if constexpr (compile_for_apple)
  {
    instance_extensions.emplace_back(
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instance_extensions.emplace_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  }

  const auto validation_layers_required =
      debug_vulkan_instance ? std::vector{"VK_LAYER_KHRONOS_validation"}
                            : std::vector<const char*>{};

  const auto instance =
      svk::Instance::Builder{}
          .application_name("GLFW-Vulkan Application")
          .engine_name("No Engine")
          .enable_instance_extensions(instance_extensions)
          .enable_validation_layers(validation_layers_required)
          .create();

  // Initialize a Vulkan surface to which the GLFW Window surface is bound.
  auto surface = kvk::Surface{instance, window};

  // List all Vulkan physical devices.
  const auto physical_devices =
      svk::PhysicalDevice::list_physical_devices(instance);

  // Find a suitable physical (GPU) device that can be used for 3D graphics
  // application.
  const auto di = std::find_if(
      physical_devices.begin(), physical_devices.end(),
      [&surface](const auto& d) {
        return d.supports_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME) &&
               !kvk::find_graphics_queue_family_indices(d).empty() &&
               !kvk::find_present_queue_family_indices(d, surface).empty();
      });

  // There must be a suitable GPU device...
  BOOST_CHECK(di != physical_devices.end());
  const auto& physical_device = *di;

  // According to:
  // https://stackoverflow.com/questions/61434615/in-vulkan-is-it-beneficial-for-the-graphics-queue-family-to-be-separate-from-th
  //
  // Using distinct queue families, namely one for the graphics operations and
  // another for the present operations, does not result in better performance.
  //
  // This is because the hardware does not expose present-only queue families...
  const auto graphics_queue_family_index =
      kvk::find_graphics_queue_family_indices(physical_device).front();
  const auto present_queue_family_index =
      kvk::find_present_queue_family_indices(physical_device, surface).front();

  // Create a logical device.
  auto device_extensions = std::vector{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  if constexpr (compile_for_apple)
    device_extensions.emplace_back("VK_KHR_portability_subset");
  const auto device = svk::Device::Builder{*di}
                          .enable_device_extensions(device_extensions)
                          .enable_queue_families({graphics_queue_family_index,
                                                  present_queue_family_index})
                          .enable_physical_device_features({})
                          .enable_validation_layers(validation_layers_required)
                          .create();
  BOOST_CHECK(static_cast<VkDevice>(device) != nullptr);

  // Now initialize the swapchain to present the rendering on screen.
  const auto swapchain =
      kvk::Swapchain{physical_device, device, surface, window};
  BOOST_CHECK(swapchain.handle != nullptr);

  // Now build the render pass.
  auto render_pass = kvk::RenderPass{};
  render_pass.create_basic_render_pass(device, swapchain.image_format);
  BOOST_CHECK(render_pass.handle != nullptr);
  BOOST_CHECK_EQUAL(render_pass.color_attachments.size(), 1u);
  BOOST_CHECK_EQUAL(render_pass.color_attachment_refs.size(),
                    render_pass.color_attachments.size());
  BOOST_CHECK_EQUAL(render_pass.subpasses.size(), 1u);
  BOOST_CHECK_EQUAL(render_pass.dependencies.size(), 1u);

  // Now build the graphics pipeline.
  namespace fs = std::filesystem;
  const auto shader_dir_path =
      fs::path(boost::unit_test::framework::master_test_suite().argv[0])
          .parent_path() /
      "test_shaders";
  const auto vshader_path = shader_dir_path / "vert.spv";
  const auto fshader_path = shader_dir_path / "frag.spv";

  const auto [w, h] = window.sizes();
  SARA_CHECK(w);
  SARA_CHECK(h);

  const auto graphics_pipeline =
      kvk::GraphicsPipeline::Builder{device, render_pass}
          .vertex_shader_path(vshader_path)
          .fragment_shader_path(fshader_path)
          .vbo_data_format<Vertex>()
          .input_assembly_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
          .viewport_sizes(static_cast<float>(w), static_cast<float>(h))
          .scissor_sizes(w, h)
          .create();
  BOOST_CHECK(graphics_pipeline.device != nullptr);
  BOOST_CHECK(graphics_pipeline.pipeline_layout != nullptr);
  BOOST_CHECK(static_cast<VkPipeline>(graphics_pipeline) != nullptr);
}
