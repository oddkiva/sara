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

  //! Pre-condition: call glfwInit() first.
  auto list_required_vulkan_extensions_from_glfw() -> std::vector<const char*>;

}  // namespace DO::Kalpana::Vulkan


namespace DO::Kalpana::Vulkan {

  //! Pre-condition: call glfwInit() first.
  //!
  //! N.B.: this does not use C++ RAII.
  //! Using RAII is not a good idea anyways because the order in which the
  //! surface object is destroyed matters.
  class Surface
  {
  public:
    Surface() = default;

    auto init(const VkInstance instance, GLFWwindow* window) -> bool
    {
      SARA_DEBUG
          << "[VK] Initializing Vulkan surface with the GLFW application...\n";
      const auto status =
          glfwCreateWindowSurface(instance, window, nullptr, &handle);
      if (status != VK_SUCCESS)
      {
        SARA_DEBUG << fmt::format("[VK] Error: failed to initilialize Vulkan "
                                  "surface! Error code: {}\n",
                                  static_cast<int>(status));
        return false;
      }

      return true;
    }

    auto destroy(const VkInstance instance) -> void
    {
      if (handle == nullptr)
        return;

      SARA_DEBUG << "[VK] Destroying Vulkan surface...\n";
      vkDestroySurfaceKHR(instance, handle, nullptr);
    }

    operator VkSurfaceKHR&()
    {
      return handle;
    }

    operator const VkSurfaceKHR&() const
    {
      return handle;
    }

  private:
    VkSurfaceKHR handle = nullptr;
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
