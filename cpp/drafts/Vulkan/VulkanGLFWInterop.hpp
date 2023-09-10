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
      if (glfwCreateWindowSurface(instance, window, nullptr, &_surface) !=
          VK_SUCCESS)
      {
        SARA_DEBUG << "[VK] Error: failed to initilialize Vulkan surface!\n";
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

  private:
    VkSurfaceKHR _surface = nullptr;
  };

}  // namespace DO::Kalpana
