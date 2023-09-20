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

#include <drafts/Vulkan/Device.hpp>
#include <drafts/Vulkan/EasyGLFW.hpp>
#include <drafts/Vulkan/GraphicsPipeline.hpp>
#include <drafts/Vulkan/Instance.hpp>
#include <drafts/Vulkan/PhysicalDevice.hpp>
#include <drafts/Vulkan/RenderPass.hpp>
#include <drafts/Vulkan/Surface.hpp>
#include <drafts/Vulkan/Swapchain.hpp>

#include <drafts/Vulkan/Geometry.hpp>

#include <DO/Sara/Defines.hpp>

#include <boost/test/unit_test.hpp>


static constexpr auto debug_vulkan_instance = true;
#if defined(__APPLE__)
static constexpr auto compile_for_apple = true;
#else
static constexpr auto compile_for_apple = false;
#endif


auto get_program_path() -> std::filesystem::path
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
                          .enable_device_features({})
                          .enable_validation_layers(validation_layers_required)
                          .create();
  BOOST_CHECK(device.handle != nullptr);

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
#if defined(__APPLE__)
  static const auto vs_path =
      "/Users/oddkiva/GitLab/oddkiva/sara-build-Debug/vert.spv";
  static const auto fs_path =
      "/Users/oddkiva/GitLab/oddkiva/sara-build-Debug/frag.spv";
#elif defined(_WIN32)
  static const auto vs_path =
      "C:/Users/David/Desktop/GitLab/sara-build-vs2022-static/vert.spv";
  static const auto fs_path =
      "C:/Users/David/Desktop/GitLab/sara-build-vs2022-static/frag.spv";
#else
  static const auto vs_path =
      "/home/david/GitLab/oddkiva/sara-build-Asan/vert.spv";
  static const auto fs_path =
      "/home/david/GitLab/oddkiva/sara-build-Asan/frag.spv";
#endif
  std::cout << vs_path << std::endl;
  std::cout << fs_path << std::endl;

  const auto [w, h] = window.sizes();
  SARA_CHECK(w);
  SARA_CHECK(h);

  const auto graphics_pipeline =
      kvk::GraphicsPipeline::Builder{device, render_pass}
          .vertex_shader_path(vs_path)
          .fragment_shader_path(fs_path)
          .vbo_data_format<Vertex>()
          .input_assembly_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
          .viewport_sizes(static_cast<float>(w), static_cast<float>(h))
          .scissor_sizes(w, h)
          .create();
  BOOST_CHECK(graphics_pipeline.device() != nullptr);
  BOOST_CHECK(graphics_pipeline.pipeline_layout() != nullptr);
  BOOST_CHECK(static_cast<VkPipeline>(graphics_pipeline) != nullptr);
}
