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

#include <drafts/Vulkan/VulkanGLFWInterop.hpp>

#include <GLFW/glfw3.h>

#include <cstdint>


namespace k = DO::Kalpana;

auto k::list_required_vulkan_extensions_from_glfw() -> std::vector<const char*>
{
  auto glfw_extension_count = std::uint32_t{};
  const char** glfw_extensions = nullptr;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

  const auto extensions = std::vector<const char*>(
      glfw_extensions, glfw_extensions + glfw_extension_count);

  return extensions;
}
