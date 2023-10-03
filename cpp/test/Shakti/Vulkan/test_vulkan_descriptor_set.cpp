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

#include <DO/Sara/Core/EigenExtension.hpp>

#include <DO/Shakti/Vulkan/Buffer.hpp>
#include <DO/Shakti/Vulkan/DescriptorPool.hpp>
#include <DO/Shakti/Vulkan/DescriptorSet.hpp>
#include <DO/Shakti/Vulkan/Device.hpp>
#include <DO/Shakti/Vulkan/DeviceMemory.hpp>
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

  // Now let's exhibit a usage code example:
  //
  // The model-view-projection matrix data needs to be sent to the vertex shader
  // in the form of a UBO.
  //
  // The UBO can be accessed by the vertex shader via a descriptor.
  //
  // We only need 1 descriptor pool.
  static constexpr auto num_pools = 1;
  // We need as many **sets** of descriptors as frames in flight.
  static constexpr auto num_frames_in_flight = 3;
  static constexpr auto& num_desc_sets = num_frames_in_flight;
  static constexpr auto& max_descriptor_count = num_frames_in_flight;

  // 1. Create UBOs and allocate a device memory for each one of them.
  struct ModelViewProjectionStack
  {
    Eigen::Matrix4f model;
    Eigen::Matrix4f view;
    Eigen::Matrix4f projection;
  };
  auto ubos = std::array<svk::Buffer, 3>{};
  auto device_mems = std::array<svk::DeviceMemory, 3>{};
  for (auto i = 0; i < 3; ++i)
  {
    ubos[i] = svk::BufferFactory{device}
                  .make_uniform_buffer<ModelViewProjectionStack>(1);
    BOOST_CHECK(static_cast<VkBuffer>(ubos[i]) != nullptr);

    device_mems[i] = svk::DeviceMemoryFactory{physical_device, device}
                         .allocate_for_uniform_buffer(ubos[i]);
    ubos[i].bind(device_mems[i], 0);

    BOOST_CHECK(static_cast<VkDeviceMemory&>(device_mems[i]) != nullptr);
  }

  // 2. Create a single descriptor pool of uniform buffer objects.
  auto desc_pool_builder = svk::DescriptorPool::Builder{device}  //
                               .pool_count(num_pools)
                               .pool_max_sets(num_desc_sets);
  // The descriptor pool can only allocate UBO descriptors.
  desc_pool_builder.pool_type(0) = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  // The cumulated number of UBO descriptors across all every descriptor sets
  // cannot exceed the following number of descriptors (3).
  desc_pool_builder.descriptor_count(0) = max_descriptor_count;

  auto desc_pool = desc_pool_builder.create();
  BOOST_CHECK(static_cast<VkDescriptorPool>(desc_pool) != nullptr);

  // 3. Create 3 descriptor sets from this pool, each one having only 1
  //    descriptor.
  //    Each descriptor set must have a layout associated to it.
  //
  // 3.a) Create one descriptor set layout for each each descriptor.
  const auto ubo_layout_binding =
      svk::DescriptorSetLayout::create_for_single_ubo(device);
  const auto desc_set_layouts =
      std::array<VkDescriptorSetLayout, num_desc_sets>{
          ubo_layout_binding,  //
          ubo_layout_binding,  //
          ubo_layout_binding   //
      };
  // 3.b) Create the 3 descriptor sets, each one of them bound with a set
  // layout, in one go.
  const auto desc_sets = svk::DescriptorSets{
      desc_set_layouts.data(),                              //
      static_cast<std::uint32_t>(desc_set_layouts.size()),  //
      desc_pool                                             //
  };
  for (auto i = 0; i < num_desc_sets; ++i)
    BOOST_CHECK(static_cast<VkDescriptorSet>(desc_sets[i]) != nullptr);

  for (auto i = 0; i < num_desc_sets; ++i)
  {
    // 4.a) Register the byte size, the type of buffer which the descriptor
    //      references to.
    auto desc_set_write = VkWriteDescriptorSet{};

    desc_set_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    desc_set_write.dstSet = desc_sets[i];
    desc_set_write.dstBinding = 0;
    desc_set_write.dstArrayElement = 0;
    desc_set_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;  // Here
    desc_set_write.descriptorCount = 1;  // Only 1 descriptor per set.

    // 4.b) Each descriptor set being a singleton, must reference to a UBO.
    auto buffer_info = VkDescriptorBufferInfo{};
    buffer_info.buffer = ubos[i];
    buffer_info.offset = 0;
    buffer_info.range = sizeof(ModelViewProjectionStack);
    desc_set_write.pBufferInfo = &buffer_info;

    // 4.c) Send this metadata to Vulkan.
    vkUpdateDescriptorSets(device, 1, &desc_set_write, 0, nullptr);
  }
}
