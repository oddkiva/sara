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
