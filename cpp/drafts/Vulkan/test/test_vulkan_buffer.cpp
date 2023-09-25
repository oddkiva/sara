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


BOOST_AUTO_TEST_CASE(test_buffer)
{
  namespace svk = DO::Shakti::Vulkan;

  const auto device = VkDevice{nullptr};
  const auto size = 0;
  const auto usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  const auto buffer = svk::Buffer{device, size, usage};

  const auto properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  const auto device_memory = svk::DeviceMemory{device, };
}
