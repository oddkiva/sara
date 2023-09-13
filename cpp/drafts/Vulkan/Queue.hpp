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

#pragma once

#include <drafts/Vulkan/Device.hpp>


namespace DO::Shakti::Vulkan {

  struct Queue
  {
    Queue() = default;

    Queue(const Device& device, const std::uint32_t queue_index)
    {
      vkGetDeviceQueue(device.handle, queue_index, 0, &handle);
    }

    VkQueue handle = nullptr;
  };

}  // namespace DO::Shakti::Vulkan
