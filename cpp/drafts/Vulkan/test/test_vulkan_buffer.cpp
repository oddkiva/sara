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

#define BOOST_TEST_MODULE "Vulkan/Buffer"

#define GLFW_INCLUDE_VULKAN

#include <drafts/Vulkan/Buffer.hpp>
#include <drafts/Vulkan/Device.hpp>
#include <drafts/Vulkan/DeviceMemory.hpp>
#include <drafts/Vulkan/Instance.hpp>
#include <drafts/Vulkan/PhysicalDevice.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Defines.hpp>

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE(test_staging_buffer)
{
  namespace svk = DO::Shakti::Vulkan;

  static constexpr auto debug_vulkan_instance = true;
#if defined(__APPLE__)
  static constexpr auto compile_for_apple = true;
#else
  static constexpr auto compile_for_apple = false;
#endif

  // Vulkan instance.
  auto instance_extensions = std::vector<const char*>{};
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
          .application_name("Vulkan Application")
          .engine_name("No Engine")
          .enable_instance_extensions(instance_extensions)
          .enable_validation_layers(validation_layers_required)
          .create();

  // List all Vulkan physical devices.
  const auto physical_devices =
      svk::PhysicalDevice::list_physical_devices(instance);

  // There must be at least one GPU device...
  BOOST_CHECK(!physical_devices.empty());
  const auto& physical_device = physical_devices.front();

  // The physical device should at least support a compute queue family.
  auto compute_queue_family_index = std::uint32_t{};
  for (auto i = 0u; i != physical_device.queue_families.size(); ++i)
  {
    if (physical_device.supports_queue_family_type(i, VK_QUEUE_COMPUTE_BIT))
    {
      compute_queue_family_index = i;
      break;
    }
  }
  BOOST_CHECK(compute_queue_family_index !=
              physical_device.queue_families.size());

  // Create a logical device.
  auto device_extensions = std::vector<const char*>{};
  if constexpr (compile_for_apple)
    device_extensions.emplace_back("VK_KHR_portability_subset");
  const auto device = svk::Device::Builder{physical_device}
                          .enable_device_extensions(device_extensions)
                          .enable_device_features({})
                          .enable_queue_families({compute_queue_family_index})
                          .enable_validation_layers(validation_layers_required)
                          .create();
  BOOST_CHECK(static_cast<VkDevice>(device) != nullptr);

  // Let's say we want to store vertex data on a device buffer.
  static constexpr auto num_vertices = 10;
  static constexpr auto vertex_byte_size = sizeof(Eigen::Vector3f);
  static constexpr auto batch_byte_size = vertex_byte_size * num_vertices;
  SARA_CHECK(vertex_byte_size);
  SARA_CHECK(batch_byte_size);

  // Our such intent is substantified by a staging buffer, which we create as
  // follows.
  const auto usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  const auto staging_buffer = svk::Buffer{device, batch_byte_size, usage};
  BOOST_CHECK(static_cast<VkBuffer>(staging_buffer) != nullptr);

  // Our hardware requirements is substantified by the memory type needed to
  // created such a buffer.
  const auto mem_reqs = staging_buffer.get_memory_requirements();
  SARA_CHECK(mem_reqs.size);
  SARA_CHECK(mem_reqs.alignment);
  SARA_CHECK(mem_reqs.memoryTypeBits);

  // The memory properties is listed precisely here:
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceMemoryProperties.html
  const auto mem_props = VkMemoryPropertyFlags{
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT  //
  };
  const auto mem_type =
      physical_device.find_memory_type(mem_reqs.memoryTypeBits,  //
                                       mem_props);
  SARA_CHECK(mem_type);

  // Finally, allocate some device memory that fulfills these such memory
  // properties.
  auto device_memory = svk::DeviceMemory{device, mem_reqs.size, mem_type};
  BOOST_CHECK(static_cast<VkDeviceMemory>(device_memory) != nullptr);

  // Bind the staging buffer to this device memory.
  staging_buffer.bind(device_memory, 0);

  // Some source vertex data.
  auto host_vertices = std::vector<float>(num_vertices * 3, 0);
  // Copy to the device memory.
  device_memory.copy_from(host_vertices.data(), host_vertices.size());

  // This will fail and Vulkan detects that we are trying to write more data
  // than what is allocated for the device memory.
  // auto host_vertices = std::vector<float>(num_vertices * 3 + 1, 0);
  // device_memory.copy_from(host_vertices.data(), host_vertices.size() + 1);
}

BOOST_AUTO_TEST_CASE(test_device_buffer)
{
  namespace svk = DO::Shakti::Vulkan;

  static constexpr auto debug_vulkan_instance = true;
#if defined(__APPLE__)
  static constexpr auto compile_for_apple = true;
#else
  static constexpr auto compile_for_apple = false;
#endif

  // Vulkan instance.
  auto instance_extensions = std::vector<const char*>{};
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
          .application_name("Vulkan Application")
          .engine_name("No Engine")
          .enable_instance_extensions(instance_extensions)
          .enable_validation_layers(validation_layers_required)
          .create();

  // List all Vulkan physical devices.
  const auto physical_devices =
      svk::PhysicalDevice::list_physical_devices(instance);

  // There must be at least one GPU device...
  BOOST_CHECK(!physical_devices.empty());
  const auto& physical_device = physical_devices.front();

  // The physical device should at least support a compute queue family.
  auto compute_queue_family_index = std::uint32_t{};
  for (auto i = 0u; i != physical_device.queue_families.size(); ++i)
  {
    if (physical_device.supports_queue_family_type(i, VK_QUEUE_COMPUTE_BIT))
    {
      compute_queue_family_index = i;
      break;
    }
  }
  BOOST_CHECK(compute_queue_family_index !=
              physical_device.queue_families.size());

  // Create a logical device.
  auto device_extensions = std::vector<const char*>{};
  if constexpr (compile_for_apple)
    device_extensions.emplace_back("VK_KHR_portability_subset");
  const auto device = svk::Device::Builder{physical_device}
                          .enable_device_extensions(device_extensions)
                          .enable_device_features({})
                          .enable_queue_families({compute_queue_family_index})
                          .enable_validation_layers(validation_layers_required)
                          .create();
  BOOST_CHECK(static_cast<VkDevice>(device) != nullptr);

  // Let's say we want to store vertex data on a device buffer.
  static constexpr auto num_vertices = 10;
  static constexpr auto vertex_byte_size = sizeof(Eigen::Vector3f);
  static constexpr auto batch_byte_size = vertex_byte_size * num_vertices;
  SARA_CHECK(vertex_byte_size);
  SARA_CHECK(batch_byte_size);

  // Our such intent is substantified by a device-only buffer, which we create
  // as follows.
  const auto usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  const auto device_buffer = svk::Buffer{device, batch_byte_size, usage};
  BOOST_CHECK(static_cast<VkBuffer>(device_buffer) != nullptr);

  // Our hardware requirements is substantified by the memory type needed to
  // created such a buffer.
  const auto mem_reqs = device_buffer.get_memory_requirements();
  SARA_CHECK(mem_reqs.size);
  SARA_CHECK(mem_reqs.alignment);
  SARA_CHECK(mem_reqs.memoryTypeBits);

  // The memory properties is listed precisely here:
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceMemoryProperties.html
  const auto mem_props =
      VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
  const auto mem_type =
      physical_device.find_memory_type(mem_reqs.memoryTypeBits,  //
                                       mem_props);
  SARA_CHECK(mem_type);

  // Finally, allocate some device memory that fulfills these such memory
  // properties.
  auto device_memory = svk::DeviceMemory{device, mem_reqs.size, mem_type};
  BOOST_CHECK(static_cast<VkDeviceMemory>(device_memory) != nullptr);

  // Bind the device buffer to this device memory.
  device_buffer.bind(device_memory, 0);
}
