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

#include <GLFW/glfw3.h>

#include <stdexcept>


namespace DO::Kalpana {

  class GLFWApplication
  {
  public:
    GLFWApplication()
      : _initialized{glfwInit() == GLFW_TRUE}
    {
      if (!_initialized)
        throw std::runtime_error{
            "Error: failed to initialize GLFW application!"};
    }

    auto init_for_vulkan_rendering() -> void
    {
      glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    }

    ~GLFWApplication()
    {
      if (_initialized)
        glfwTerminate();
    }

  private:
    bool _initialized = false;
  };

}  // namespace DO::Kalpana
