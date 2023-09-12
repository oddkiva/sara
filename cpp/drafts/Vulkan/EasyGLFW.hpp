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

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <GLFW/glfw3.h>

#include <fmt/format.h>

#include <stdexcept>
#include <string>


namespace DO::Kalpana::GLFW {

  class Application
  {
  public:
    Application()
      : _initialized{glfwInit() == GLFW_TRUE}
    {
      if (!_initialized)
        throw std::runtime_error{
            "[GLFW] Error: failed to initialize GLFW application!"};
      SARA_DEBUG << "[GLFW] Initialized GLFW application!\n";
    }

    auto init_for_vulkan_rendering() -> void
    {
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    }

    ~Application()
    {
      if (_initialized)
      {
        SARA_DEBUG << "[GLFW] Terminating GLFW application!\n";
        glfwTerminate();
        _initialized = false;
      }
    }

  private:
    bool _initialized = false;
  };

  class Window
  {
  public:
    using Handle = GLFWwindow*;

    Window() = default;

    Window(const int w, const int h, const std::string_view& title)
    {
      _handle = glfwCreateWindow(w, h, title.data(), nullptr, nullptr);
      SARA_DEBUG << fmt::format("[GLFW] Created GLFW window: {}\n",
                                fmt::ptr(_handle));
    }

    ~Window()
    {
      if (_handle == nullptr)
        return;

      SARA_DEBUG << fmt::format("[GLFW] Destroying GLFW window: {}\n",
                                fmt::ptr(_handle));
      glfwDestroyWindow(_handle);
    }

    auto sizes() const -> std::array<int, 2>
    {
      auto sz = std::array<int, 2>{};
      glfwGetWindowSize(_handle, &sz[0], &sz[1]);
      return sz;
    }

    operator Handle&()
    {
      return _handle;
    }

    operator Handle() const
    {
      return _handle;
    }

  private:
    GLFWwindow* _handle = nullptr;
  };

}  // namespace DO::Kalpana::GLFW
