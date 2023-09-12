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

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <drafts/Vulkan/Instance.hpp>
#include <drafts/Vulkan/PhysicalDevice.hpp>


namespace DO::Kalpana::Vulkan {

  //! Pre-condition: call glfwInit() first or instantiate a
  //! `DO::Kalpana::GLFW::Application` object.
  class Surface
  {
  public:
    Surface() = default;

    Surface(const VkInstance instance, GLFWwindow* window)
      : _instance{instance}
    {
      const auto status =
          glfwCreateWindowSurface(_instance, window, nullptr, &_handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Error: failed to initilialize Vulkan "
                        "surface! Error code: {}\n",
                        static_cast<int>(status))};
      SARA_DEBUG << fmt::format("[VK] Initialized Vulkan surface: {}\n",
                                fmt::ptr(_handle));
    }

    ~Surface()
    {
      if (_handle == nullptr || _instance == nullptr)
        return;

      SARA_DEBUG << fmt::format("[VK] Destroying Vulkan surface: {}\n",
                                fmt::ptr(_handle));
      vkDestroySurfaceKHR(_instance, _handle, nullptr);
    }

    operator VkSurfaceKHR&()
    {
      return _handle;
    }

    operator const VkSurfaceKHR&() const
    {
      return _handle;
    }

    static auto list_required_instance_extensions_from_glfw()
        -> std::vector<const char*>;

  private:
    VkSurfaceKHR _handle = nullptr;
    VkInstance _instance = nullptr;
  };


  inline auto
  find_graphics_queue_family_indices(const Shakti::Vulkan::PhysicalDevice& d)
      -> std::vector<std::uint32_t>
  {
    auto indices = std::vector<std::uint32_t>{};
    for (auto i = std::uint32_t{}; i != d.queue_families.size(); ++i)
      if (d.supports_queue_family_type(i, VK_QUEUE_GRAPHICS_BIT))
        indices.emplace_back(i);
    return indices;
  }

  inline auto
  find_present_queue_family_indices(const Shakti::Vulkan::PhysicalDevice& d,
                                    const Surface& s)
      -> std::vector<std::uint32_t>
  {
    auto indices = std::vector<std::uint32_t>{};
    for (auto i = std::uint32_t{}; i != d.queue_families.size(); ++i)
      if (d.supports_surface_presentation(i, s))
        indices.emplace_back(i);
    return indices;
  }

}  // namespace DO::Kalpana::Vulkan
