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

#include <drafts/Vulkan/Instance.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <GLFW/glfw3.h>
#include <vulkan/vulkan_core.h>


namespace DO::Kalpana {

  //! Pre-condition: call glfwInit() first.
  auto list_required_vulkan_extensions_from_glfw() -> std::vector<const char*>;

}  // namespace DO::Kalpana


namespace DO::Kalpana {

  //! Pre-condition: call glfwInit() first.
  //!
  //! N.B.: this does not use C++ RAII.
  class Surface
  {
  public:
    Surface() = default;

    auto init(const VkInstance instance, GLFWwindow* window) -> bool
    {
      SARA_DEBUG
          << "[VK] Initializing Vulkan surface with the GLFW application...\n";
      const auto status =
          glfwCreateWindowSurface(instance, window, nullptr, &_surface);
      if (status != VK_SUCCESS)
      {
        SARA_DEBUG << fmt::format("[VK] Error: failed to initilialize Vulkan "
                                  "surface! Error code: {}\n",
                                  status);
        return false;
      }

      return true;
    }

    auto destroy(const VkInstance instance) -> void
    {
      if (_surface == nullptr)
        return;

      SARA_DEBUG << "[VK] Destroying Vulkan surface...\n";
      vkDestroySurfaceKHR(instance, _surface, nullptr);
    }

    operator VkSurfaceKHR&()
    {
      return _surface;
    }

    operator const VkSurfaceKHR&() const
    {
      return _surface;
    }

  private:
    VkSurfaceKHR _surface = nullptr;
  };

}  // namespace DO::Kalpana
