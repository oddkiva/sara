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

#define BOOST_TEST_MODULE "Vulkan/Descriptor Pool and Set"

#include <DO/Shakti/Vulkan/DescriptorPool.hpp>
#include <DO/Shakti/Vulkan/DescriptorSet.hpp>
#include <DO/Shakti/Vulkan/Device.hpp>
#include <DO/Shakti/Vulkan/Instance.hpp>
#include <DO/Shakti/Vulkan/PhysicalDevice.hpp>

#include <boost/test/unit_test.hpp>


static constexpr auto debug_vulkan_instance = true;
#if defined(__APPLE__)
static constexpr auto compile_for_apple = true;
#else
static constexpr auto compile_for_apple = false;
#endif


BOOST_AUTO_TEST_CASE(test_device)
{
  namespace svk = DO::Shakti::Vulkan;

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
  auto device_extensions = std::vector{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  if constexpr (compile_for_apple)
    device_extensions.emplace_back("VK_KHR_portability_subset");
  const auto device = svk::Device::Builder{physical_device}
                          .enable_device_extensions(device_extensions)
                          .enable_queue_families({compute_queue_family_index})
                          .enable_device_features({})
                          .enable_validation_layers(validation_layers_required)
                          .create();
  BOOST_CHECK(static_cast<VkDevice>(device) != nullptr);

  const auto desc_pool = svk::DescriptorPool{device};
  BOOST_CHECK(static_cast<VkDescriptorPool>(desc_pool) != nullptr);

  const auto desc_set = svk::DescriptorSet{1, desc_pool};
  BOOST_CHECK(static_cast<VkDescriptorSet>(desc_set) != nullptr);
}
