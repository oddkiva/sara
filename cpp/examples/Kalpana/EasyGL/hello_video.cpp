// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Kalpana/EasyGL.hpp>
#include <DO/Kalpana/EasyGL/Objects/ImageTexture.hpp>
#include <DO/Kalpana/EasyGL/Objects/Quad.hpp>
#include <DO/Kalpana/EasyGL/Renderer/ImageTextureRenderer.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/VideoIO.hpp>

#ifdef _WIN32
#  include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <filesystem>


namespace kgl = DO::Kalpana::GL;
namespace sara = DO::Sara;
namespace fs = std::filesystem;


inline auto init_glfw_boilerplate() -> void
{
  // Initialize the windows manager.
  if (!glfwInit())
    throw std::runtime_error{"Error: failed to initialize GLFW!"};

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
}

inline auto init_glew_boilerplate() -> void
{
#ifndef __APPLE__
  // Initialize GLEW.
  const auto err = glewInit();
  if (err != GLEW_OK)
    throw std::runtime_error{sara::format(
        "Error: failed to initialize GLEW: %s", glewGetErrorString(err))};
#endif
}

inline auto create_glfw_window(const Eigen::Vector2i& sizes,
                               const std::string& title) -> GLFWwindow*
{
  auto window = glfwCreateWindow(sizes.x(), sizes.y(),  //
                                 title.c_str(),         //
                                 nullptr, nullptr);
  return window;
}

struct SingleWindowApp
{
  SingleWindowApp(const Eigen::Vector2i& sizes, const std::string& title)
  {
    // Init GLFW.
    init_glfw_boilerplate();

    // Create a GLFW window.
    _window = create_glfw_window(sizes, title);
    glfwMakeContextCurrent(_window);

    // Init OpenGL extensions.
    init_glew_boilerplate();
  }

  //! @brief Note: RAII does not work on OpenGL applications.
  //!
  //! So the destructor gets a default implementation and we neeed to explicitly
  //! call the terminate method.
  ~SingleWindowApp() = default;

  auto open_video(const fs::path& video_path) -> void
  {
    _video_stream.open(video_path.string());
  }

  auto run() -> void
  {
    // Display image.
    glfwSwapInterval(1);
    while (!glfwWindowShouldClose(_window))
    {
      glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      glfwSwapBuffers(_window);
      glfwPollEvents();
    }
  }

  auto terminate() -> void
  {
    if (_window != nullptr)
      glfwDestroyWindow(_window);
    glfwTerminate();
  }

  GLFWwindow* _window = nullptr;

  // Our video stream.
  sara::VideoStream _video_stream;
  // What: our image texture.
  kgl::ImageTexture _image_texture;
  // Where: where to show our image texture.
  kgl::Quad _quad;
};


auto main(int const argc, char** const argv) -> int
{
  if (argc < 2)
  {
    std::cout << "Usage: " << argv[0] << " VIDEO_FILEPATH" << std::endl;
    return EXIT_FAILURE;
  }

  auto app = SingleWindowApp({800, 600}, "Hello Video");
  app.open_video(fs::path{argv[1]});
  app.run();

  // Explicitly terminate the OpenGL app because RAII does not work with OpenGL.
  app.terminate();

  return EXIT_SUCCESS;
}
