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

#include <drafts/Vulkan/Buffer.hpp>
#include <drafts/Vulkan/Device.hpp>
#include <drafts/Vulkan/DeviceMemory.hpp>
#include <drafts/Vulkan/Instance.hpp>
#include <drafts/Vulkan/PhysicalDevice.hpp>

#include <DO/Sara/Defines.hpp>

#include <boost/test/unit_test.hpp>


auto find_memory_type(const VkPhysicalDevice physical_device,
                      const std::uint32_t type_filter,
                      VkMemoryPropertyFlags properties) -> std::uint32_t
{
  auto mem_properties = VkPhysicalDeviceMemoryProperties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

  for (auto i = 0u; i < mem_properties.memoryTypeCount; ++i)
  {
    if ((type_filter & (1 << i)) &&
        (mem_properties.memoryTypes[i].propertyFlags & properties) ==
            properties)
      return i;
  }

  throw std::runtime_error("failed to find suitable memory type!");
}


BOOST_AUTO_TEST_CASE(test_buffer)
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

  // There must be a suitable GPU device...
  BOOST_CHECK(!physical_devices.empty());
  const auto& physical_device = physical_devices.front();


  // Create a logical device.
  auto device_extensions = std::vector{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  if constexpr (compile_for_apple)
    device_extensions.emplace_back("VK_KHR_portability_subset");
  const auto device = svk::Device::Builder{physical_device}
                          .enable_device_extensions(device_extensions)
                          .enable_device_features({})
                          .enable_validation_layers(validation_layers_required)
                          .create();
  BOOST_CHECK(device.handle != nullptr);

  const auto size = 1024u;  // bytes
  const auto usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  const auto buffer = svk::Buffer{device.handle, size, usage};
  BOOST_CHECK(static_cast<VkBuffer>(buffer) != nullptr);
  const auto mem_reqs = buffer.get_memory_requirements();
  SARA_CHECK(mem_reqs.size);
  SARA_CHECK(mem_reqs.alignment);
  SARA_CHECK(mem_reqs.memoryTypeBits);

  const auto properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  const auto device_memory = svk::DeviceMemory{device.handle, size, {}};
  BOOST_CHECK(static_cast<VkDeviceMemory>(device_memory) != nullptr);

  mem_po
}
