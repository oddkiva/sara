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

#include "DO/Shakti/Vulkan/CommandBuffer.hpp"
#include "DO/Shakti/Vulkan/CommandPool.hpp"
#include "DO/Shakti/Vulkan/Queue.hpp"
#include <vulkan/vulkan_core.h>
#define BOOST_TEST_MODULE "Vulkan/Image"

#include <DO/Shakti/Vulkan/Buffer.hpp>
#include <DO/Shakti/Vulkan/Device.hpp>
#include <DO/Shakti/Vulkan/DeviceMemory.hpp>
#include <DO/Shakti/Vulkan/Image.hpp>
#include <DO/Shakti/Vulkan/ImageView.hpp>
#include <DO/Shakti/Vulkan/Instance.hpp>
#include <DO/Shakti/Vulkan/PhysicalDevice.hpp>

#include <boost/test/unit_test.hpp>


static constexpr auto debug_vulkan_instance = true;
#if defined(__APPLE__)
static constexpr auto compile_for_apple = true;
#else
static constexpr auto compile_for_apple = false;
#endif


BOOST_AUTO_TEST_CASE(test_image)
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
  auto device_extensions = std::vector<const char*>{};
  if constexpr (compile_for_apple)
    device_extensions.emplace_back("VK_KHR_portability_subset");
  const auto device = svk::Device::Builder{physical_device}
                          .enable_device_extensions(device_extensions)
                          .enable_queue_families({compute_queue_family_index})
                          .enable_device_features({})
                          .enable_validation_layers(validation_layers_required)
                          .create();
  BOOST_CHECK(static_cast<VkDevice>(device) != nullptr);

  static constexpr auto width = 800u;
  static constexpr auto height = 600u;

  // Create a staging buffer.
  const auto staging_image_buffer =
      svk::BufferFactory{device}  //
          .make_staging_buffer<std::uint32_t>(width * height);
  BOOST_CHECK(static_cast<VkBuffer>(staging_image_buffer) != nullptr);
  const auto staging_image_dmem =
      svk::DeviceMemoryFactory{physical_device, device}
          .allocate_for_staging_buffer(staging_image_buffer);
  BOOST_CHECK(static_cast<VkDeviceMemory>(staging_image_dmem) != nullptr);
  staging_image_buffer.bind(staging_image_dmem, 0);

  // Create an image.
  const auto image =
      svk::Image::Builder{device}
          .sizes(VkExtent2D{width, height})
          .format(VK_FORMAT_R8G8B8A8_SRGB)
          .tiling(VK_IMAGE_TILING_OPTIMAL)
          // .initial_layout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
          .usage(VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)
          .create();
  BOOST_CHECK(static_cast<VkImage>(image) != nullptr);

  // We must allocate device memory for the image before creating an image view.
  const auto image_dmem = svk::DeviceMemoryFactory{physical_device, device}
                              .allocate_for_device_image(image);
  BOOST_CHECK(static_cast<VkDeviceMemory>(image_dmem) != nullptr);
  image.bind(image_dmem, 0);

  // Create an image view.
  const auto image_view = svk::ImageView::Builder{device}
                              .image(image)
                              .format(VK_FORMAT_R8G8B8A8_SRGB)
                              .aspect_mask(VK_IMAGE_ASPECT_COLOR_BIT)
                              .create();
  BOOST_CHECK(static_cast<VkImageView>(image_view) != nullptr);


  // Copy from staging buffer object to image object.
  auto compute_queue = svk::Queue{device, compute_queue_family_index};

  // Command buffers.
  const auto cmd_pool = svk::CommandPool{device, compute_queue_family_index};
  auto cmd_bufs = svk::CommandBufferSequence{1, device, cmd_pool};
  const auto& cmd_buf = cmd_bufs[0];

  svk::record_copy_buffer_to_image(staging_image_buffer, image, cmd_buf);

  compute_queue.submit_copy_commands(cmd_bufs);
  compute_queue.wait();

  cmd_bufs.clear();
}
