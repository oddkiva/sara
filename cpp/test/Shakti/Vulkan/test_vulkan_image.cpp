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

#define BOOST_TEST_MODULE "Vulkan/Image"

#include <DO/Sara/Core/Image.hpp>

#include <DO/Shakti/Vulkan/Buffer.hpp>
#include <DO/Shakti/Vulkan/CommandBuffer.hpp>
#include <DO/Shakti/Vulkan/CommandPool.hpp>
#include <DO/Shakti/Vulkan/DescriptorPool.hpp>
#include <DO/Shakti/Vulkan/DescriptorSet.hpp>
#include <DO/Shakti/Vulkan/Device.hpp>
#include <DO/Shakti/Vulkan/DeviceMemory.hpp>
#include <DO/Shakti/Vulkan/Image.hpp>
#include <DO/Shakti/Vulkan/ImageView.hpp>
#include <DO/Shakti/Vulkan/Instance.hpp>
#include <DO/Shakti/Vulkan/PhysicalDevice.hpp>
#include <DO/Shakti/Vulkan/Queue.hpp>
#include <DO/Shakti/Vulkan/Sampler.hpp>

#include <boost/test/unit_test.hpp>


static constexpr auto debug_vulkan_instance = true;
#if defined(__APPLE__)
static constexpr auto compile_for_apple = true;
#else
static constexpr auto compile_for_apple = false;
#endif


BOOST_AUTO_TEST_CASE(test_image)
{
  namespace sara = DO::Sara;
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
  // Allow anisotropic sampling.
  auto physical_device_features = VkPhysicalDeviceFeatures{};
  physical_device_features.samplerAnisotropy = VK_TRUE;
  const auto device = svk::Device::Builder{physical_device}
                          .enable_device_extensions(device_extensions)
                          .enable_queue_families({compute_queue_family_index})
                          .enable_device_features(physical_device_features)
                          .enable_validation_layers(validation_layers_required)
                          .create();
  BOOST_CHECK(static_cast<VkDevice>(device) != VK_NULL_HANDLE);


  // ======================================================================== //
  // TODO: SPECIFY THE GRAPHICS PIPELINE LAYOUT.
  //
  // 1. Create the description set layout that the shaders need.
  const auto desc_set_layout = svk::DescriptorSetLayout::Builder{device}
                                   .push_sampler_layout_binding()
                                   .create();
  // 2. Hook this to the graphics pipeline layout data structure.


  // ======================================================================== //
  // ALLOCATE IMAGE RESOURCE ON VULKAN SIDE.
  //
  // Make some constants.
  static constexpr auto width = 800;
  static constexpr auto height = 600;
  static const auto white = sara::Rgba8{255, 255, 255, 255};
  static const auto black = sara::Rgba8{0, 0, 0, 0};

  // Create a white image on the host side.
  auto host_image = sara::Image<sara::Rgba8>{width, height};
  host_image.flat_array().fill(white);

  // Create a staging buffer on the Vulkan device side.
  const auto staging_image_buffer =
      svk::BufferFactory{device}  //
          .make_staging_buffer<std::uint32_t>(host_image.size());
  BOOST_CHECK(static_cast<VkBuffer>(staging_image_buffer) != VK_NULL_HANDLE);
  const auto staging_image_dmem =
      svk::DeviceMemoryFactory{physical_device, device}
          .allocate_for_staging_buffer(staging_image_buffer);
  BOOST_CHECK(static_cast<VkDeviceMemory>(staging_image_dmem) !=
              VK_NULL_HANDLE);
  staging_image_buffer.bind(staging_image_dmem, 0);

  // Copy the image data from the host side to the device side.
  staging_image_dmem.copy_from(host_image.data(), host_image.size());

  // Double-check the content of the staging buffer.
  {
    const auto staging_image_ptr_2 =
        staging_image_dmem.map_memory<sara::Rgba8>(host_image.size());
    auto staging_image_dmem_copy_on_host =
        sara::Image<sara::Rgba8>{width, height};
    staging_image_dmem_copy_on_host.flat_array().fill(black);
    std::memcpy(staging_image_dmem_copy_on_host.data(), staging_image_ptr_2,
                staging_image_dmem.size());
    staging_image_dmem.unmap_memory();
    BOOST_CHECK(host_image == staging_image_dmem_copy_on_host);
  }

  // Create an image on the device side.
  const auto image =
      svk::Image::Builder{device}
          .sizes(VkExtent2D{width, height})
          .format(VK_FORMAT_R8G8B8A8_SRGB)
          .tiling(VK_IMAGE_TILING_OPTIMAL)
          .usage(VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)
          .create();
  BOOST_CHECK(static_cast<VkImage>(image) != VK_NULL_HANDLE);

  // We must allocate a device memory for the image before creating an image
  // view.
  const auto image_dmem = svk::DeviceMemoryFactory{physical_device, device}
                              .allocate_for_device_image(image);
  BOOST_CHECK(static_cast<VkDeviceMemory>(image_dmem) != VK_NULL_HANDLE);
  image.bind(image_dmem, 0);

  // Copy the data from the staging buffer object to the image object.
  //
  // So first create a queue, a command pool, and a command buffer to submit
  // operations on Vulkan side.
  const auto compute_queue = svk::Queue{device, compute_queue_family_index};
  const auto cmd_pool = svk::CommandPool{device, compute_queue_family_index};
  auto cmd_bufs = svk::CommandBufferSequence{1, device, cmd_pool};
  const auto& cmd_buf = cmd_bufs[0];

  // Allow the image to be used as a copy destination resource.
  svk::record_image_layout_transition(image, VK_IMAGE_LAYOUT_UNDEFINED,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      cmd_buf);
  compute_queue.submit_commands(cmd_bufs);
  compute_queue.wait();

  // Copy the data from the staging buffer to the device image.
  cmd_bufs.reset(0);
  svk::record_copy_buffer_to_image(staging_image_buffer, image, cmd_buf);
  compute_queue.submit_commands(cmd_bufs);
  compute_queue.wait();

  // Finally tell Vulkan that the image can only be used a read-only resource
  // from a shader now on.
  cmd_bufs.reset(0);
  svk::record_image_layout_transition(
      image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, cmd_buf);
  compute_queue.submit_commands(cmd_bufs);
  compute_queue.wait();
  // Optional since the destructor calls it: but this helped to fix a bug in the
  // implementation of the `clear` method.
  cmd_bufs.clear();

  // To use the image resource from a shader:
  // 1. Create an image view
  // 2. Create an image sampler
  // 3. Add a DescriptorSetLayout for the image sampler.
  const auto image_view = svk::ImageView::Builder{device}
                              .image(image)
                              .format(VK_FORMAT_R8G8B8A8_SRGB)
                              .aspect_mask(VK_IMAGE_ASPECT_COLOR_BIT)
                              .create();
  BOOST_CHECK(static_cast<VkImageView>(image_view) != VK_NULL_HANDLE);

  const auto image_sampler =
      svk::Sampler::Builder{physical_device, device}.create();
  BOOST_CHECK(static_cast<VkSampler>(image_sampler) != VK_NULL_HANDLE);


  // ======================================================================== //
  // ALLOCATE RENDER RESOURCES ON VULKAN (descriptor pool, sets and so on.)
  //
  // 2. A set of descriptors allocated by a descriptor pool.
  //
  // We only need 1 pool of image sampler descriptors.
  static constexpr auto num_pools = 1;
  static constexpr auto num_frames_in_flight = 2;
  auto desc_pool_builder = svk::DescriptorPool::Builder{device}
                               .pool_count(num_pools)  //
                               .pool_max_sets(num_frames_in_flight);
  desc_pool_builder.descriptor_count(0) = num_frames_in_flight;
  desc_pool_builder.pool_type(0) = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

  auto desc_pool = desc_pool_builder.create();
  BOOST_CHECK(static_cast<VkDescriptorPool>(desc_pool) != VK_NULL_HANDLE);

  const auto desc_set_layouts = std::vector<VkDescriptorSetLayout>(
      num_frames_in_flight,
      static_cast<VkDescriptorSetLayout>(desc_set_layout));

  // We create num_frames_in_flight sets of descriptors.
  auto desc_sets = svk::DescriptorSets{
      desc_set_layouts.data(),                              //
      static_cast<std::uint32_t>(desc_set_layouts.size()),  //
      desc_pool                                             //
  };


  // We describe each sets of descriptors.
  auto image_info = VkDescriptorImageInfo{};
  image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  image_info.imageView = image_view;
  image_info.sampler = image_sampler;

  auto desc_write = VkWriteDescriptorSet{};
  desc_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  desc_write.dstSet = desc_sets[0];
  desc_write.dstBinding = 0;
  desc_write.dstArrayElement = 0;
  desc_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  desc_write.descriptorCount = 1;
  desc_write.pImageInfo = &image_info;
  vkUpdateDescriptorSets(device, 1, &desc_write, 0, nullptr);
}
