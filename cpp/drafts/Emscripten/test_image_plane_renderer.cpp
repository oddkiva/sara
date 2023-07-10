// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#include <DO/Kalpana/EasyGL.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>

#include <filesystem>
#include <iostream>
#include <map>
#include <memory>

#include <GLFW/glfw3.h>

#ifdef __EMSCRIPTEN__
#  include <emscripten/emscripten.h>
#  define GLFW_INCLUDE_ES3
#endif

#include "ImagePlaneRenderer.hpp"


namespace fs = std::filesystem;
namespace sara = DO::Sara;


class GLFWApp
{
public:
  GLFWApp(const Eigen::Vector2i& sizes, const std::string& title)
    : _window_sizes{sizes}
  {
    // In the constructor, the order in which functions are called matters.
    //
    // 1. Init GLFW.
    if (glfwInit() != GL_TRUE)
      throw std::runtime_error{"Failed to initialize GLFW!"};

    // 2. Specify the OpenGL version we want to use.
    //    OpenGL core version 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // 3. Create a GLFW window.
    _window = glfwCreateWindow(sizes.x(), sizes.y(),  //
                               title.c_str(),         //
                               nullptr, nullptr);
    if (!_window)
    {
      glfwTerminate();
      throw std::runtime_error{"Failed to create window!"};
    }

    // clang-format off
// #ifdef __APPLE__
//   // GL 3.2 + GLSL 150
//   MyGLFW::glsl_version = "#version 150";
//   glfwWindowHint(  // required on Mac OS
//       GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
// #elif __linux__
//   // GL 3.2 + GLSL 150
//   glsl_version = "#version 150";
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
// #elif _WIN32
//   // GL 3.0 + GLSL 130
//   glsl_version = "#version 130";
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
// #endif
    // clang-format on

#ifdef _WIN32
    // if it's a HighDPI monitor, try to scale everything
    const auto monitor = glfwGetPrimaryMonitor();
    float xscale, yscale;
    glfwGetMonitorContentScale(monitor, &xscale, &yscale);
    if (xscale > 1 || yscale > 1)
    {
      high_dpi_scale_factor = static_cast<int>(xscale);
      glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    }
#elif __APPLE__
    // to prevent 1200x800 from becoming 2400x1600
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif

    // 4. Tell that the OpenGL rendering will be done on this window surface.
    glfwMakeContextCurrent(_window);

#ifndef __EMSCRIPTEN__
    // 5. Load GLEW.
    //    This is important as any attempt at calling OpenGL functions will
    //    result in a runtime crash.
    const auto err = glewInit();
    if (err != GLEW_OK)
      throw std::runtime_error{"Error: failed to initialize GLEW!"};
#endif
  }

  auto initialize(const fs::path& program_dir_path) -> void
  {
#ifndef __EMSCRIPTEN__
    _program_dir_path = program_dir_path;
#endif

    initialize_image_textures();

    // Specific rendering options.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
  }

  auto run() -> void
  {
#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(render_frame, 0, 1);
#else
    while (!glfwWindowShouldClose(_window))
      render_frame();
#endif
  }

  auto terminate() -> void
  {
    cleanup_gl_objects();
    glfwTerminate();
  }

private:
  auto render_frame() -> void
  {
    glViewport(0, 0, _window_sizes.x(), _window_sizes.y());

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const auto& image_textures = _image_plane_renderer._textures;
    for (const auto& image_texture : image_textures)
      _image_plane_renderer.render(image_texture);

    glfwSwapBuffers(_window);
    glfwPollEvents();
  }

  auto initialize_image_textures() -> void
  {
    _image_plane_renderer.initialize();

#ifdef __EMSCRIPTEN__
    const auto images = std::array<sara::Image<sara::Rgb8>, 2>{
        sara::imread<sara::Rgb8>("assets/image-omni.png"),
        sara::imread<sara::Rgb8>("assets/image-pinhole.png"),
    };
#elif defined _WIN32
    auto images = std::array<sara::Image<sara::Rgb8>, 2>{};
    for (auto i = 0; i < 2; ++i)
      images[i] = sara::resize(
          sara::imread<sara::Rgb8>(
              "C:/Users/David/Desktop/GitLab/sara/data/stinkbug.png"),
          {1920, 1080});
#else
    const auto images = std::array<sara::Image<sara::Rgb8>, 2>{
        sara::imread<sara::Rgb8>(
            (_program_dir_path / "assets/image-omni.png").string()),
        sara::imread<sara::Rgb8>(
            (_program_dir_path / "assets/image-pinhole.png").string())  //
    };
#endif

    auto& image_textures = _image_plane_renderer._textures;
    image_textures.resize(2);
    for (auto i = 0; i < 2; ++i)
    {
      // Transfer the CPU image data to the GPU texture.
      const auto texture_unit = i;
      image_textures[i].set_texture(images[i], texture_unit);

      // Geometry
      image_textures[i]._model_view.setIdentity();
      image_textures[i]._projection.setIdentity();
      // Dummy transformation just to make things to see.
      image_textures[i]._model_view(0, 3) += i * 0.5f;
      image_textures[i]._model_view(2, 3) -= i * 0.5f;
    }
  }

  auto cleanup_gl_objects() -> void
  {
    // Destroy the shaders and quad geometry data.
    _image_plane_renderer.destroy_gl_objects();

    // Destroy the image textures.
    auto& image_textures = _image_plane_renderer._textures;
    for (auto i = 0u; i < image_textures.size(); ++i)
      image_textures[i].destroy();

    // Clear the list of image textures.
    image_textures.clear();
  }

private:
  GLFWwindow* _window = nullptr;
  Eigen::Vector2i _window_sizes = Eigen::Vector2i::Zero();
  fs::path _program_dir_path;

  ImagePlaneRenderer _image_plane_renderer;
};

int main(int, [[maybe_unused]] char** argv)
{
  try
  {
    auto app = GLFWApp{{800, 600}, "Image Plane Renderer"};
    // app.initialize(fs::path{argv[0]}.parent_path());
    app.initialize(fs::path{"/home/david/GitLab/DO-CV"} / "sara" / "cpp" /
                   "drafts" / "Emscripten");
    app.run();
    app.terminate();
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
