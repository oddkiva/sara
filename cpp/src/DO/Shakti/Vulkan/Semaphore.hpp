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

#include <vulkan/vulkan_core.h>

#include <fmt/format.h>

#include <algorithm>


namespace DO::Shakti::Vulkan {

  class Semaphore
  {
  public:
    Semaphore() = default;

    explicit Semaphore(VkDevice device)
      : _device{device}
    {
      auto create_info = VkSemaphoreCreateInfo{};
      create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      const auto status =
          vkCreateSemaphore(device, &create_info, nullptr, &_handle);

      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Failed to create semaphore! Error code: {}",
                        static_cast<int>(status))};
    }

    Semaphore(const Semaphore&) = delete;

    Semaphore(Semaphore&& other)
    {
      swap(other);
    }

    ~Semaphore()
    {
      if (_device == nullptr || _handle == nullptr)
        return;
      vkDestroySemaphore(_device, _handle, nullptr);
    }

    auto operator=(const Semaphore&) -> Semaphore& = delete;

    auto operator=(Semaphore&& other) -> Semaphore&
    {
      swap(other);
      return *this;
    }

    operator VkSemaphore&()
    {
      return _handle;
    }

    operator VkSemaphore() const
    {
      return _handle;
    }

    auto swap(Semaphore& other) -> void
    {
      std::swap(_device, other._device);
      std::swap(_handle, other._handle);
    }

  private:
    VkDevice _device = nullptr;
    VkSemaphore _handle = nullptr;
  };

}  // namespace DO::Shakti::Vulkan
