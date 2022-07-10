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

#include <drafts/OpenCL/GL.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>

#include <iostream>
#include <map>
#include <memory>

#ifdef __EMSCRIPTEN__
#  include <emscripten/emscripten.h>
#  define GLFW_INCLUDE_ES3
#  include <filesystem>
#else
#  include <boost/filesystem.hpp>
#endif

#include "MyGLFW.hpp"

#include "ImagePlaneRenderer.hpp"


#ifdef __EMSCRIPTEN__
namespace fs = std::filesystem;
#else
namespace fs = boost::filesystem;
#endif
namespace sara = DO::Sara;


#ifndef __EMSCRIPTEN__
static auto program_dir_path = fs::path{};
#endif


auto render_frame() -> void
{
  glViewport(0, 0, MyGLFW::width, MyGLFW::height);

  glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  auto& image_plane_renderer = ImagePlaneRenderer::instance();
  const auto& image_textures = image_plane_renderer._textures;
  for (const auto& image_texture : image_textures)
    image_plane_renderer.render(image_texture);

  glfwSwapBuffers(MyGLFW::window);
  glfwPollEvents();
}

auto initialize_image_textures()
{
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
          (program_dir_path / "assets/image-omni.png").string()),
      sara::imread<sara::Rgb8>(
          (program_dir_path / "assets/image-pinhole.png").string())};
#endif

  auto& image_plane_renderer = ImagePlaneRenderer::instance();
  auto& image_textures = image_plane_renderer._textures;
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
  auto& image_plane_renderer = ImagePlaneRenderer::instance();
  image_plane_renderer.destroy_gl_objects();

  // Destroy the image textures.
  auto& image_textures = image_plane_renderer._textures;
  for (auto i = 0u; i < image_textures.size(); ++i)
    image_textures[i].destroy();

  // Clear the list of image textures.
  image_textures.clear();
}

int main(int, [[maybe_unused]] char** argv)
{
  try
  {
    if (!MyGLFW::initialize())
      return EXIT_FAILURE;

#ifndef __EMSCRIPTEN__
    program_dir_path = fs::path{argv[0]}.parent_path();
#endif


#ifndef __EMSCRIPTEN__
    glewInit();
#endif

    auto& image_plane_renderer = ImagePlaneRenderer::instance();
    image_plane_renderer.initialize();
    initialize_image_textures();

    // Specific rendering options.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(render_frame, 0, 1);
#else
    while (!glfwWindowShouldClose(MyGLFW::window))
      render_frame();
#endif

    cleanup_gl_objects();

    glfwTerminate();
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
