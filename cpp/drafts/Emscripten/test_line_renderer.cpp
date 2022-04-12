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

#include <DO/Sara/ImageIO.hpp>

#ifdef EMSCRIPTEN
#  include <emscripten/emscripten.h>
#  define GLFW_INCLUDE_ES3
#endif

#include "MyGLFW.hpp"

#include "ImagePlaneRenderer.hpp"
#include "LineRenderer.hpp"


namespace sara = DO::Sara;


auto render_frame() -> void
{
  glViewport(0, 0, MyGLFW::width, MyGLFW::height);

  glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  auto& image_plane_renderer = ImagePlaneRenderer::instance();
  auto& line_renderer = LineRenderer::instance();

  const auto& image_textures = image_plane_renderer._textures;
  const auto& lines = line_renderer._lines;

  for (auto i = 0u; i < image_textures.size(); ++i)
    image_plane_renderer.render(image_textures[i]);

  for (auto i = 0u; i < image_textures.size(); ++i)
    line_renderer.render(image_textures[i], lines[i]);

  glfwSwapBuffers(MyGLFW::window);
  glfwPollEvents();
}

auto initialize_image_textures()
{
  const auto images = std::array<sara::Image<sara::Rgb8>, 2>{
      sara::imread<sara::Rgb8>("assets/image-omni.png"),
      sara::imread<sara::Rgb8>("assets/image-pinhole.png"),
  };

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

auto initialize_lines()
{
  auto& line_renderer = LineRenderer::instance();
  auto& gl_lines = line_renderer._lines;
  gl_lines.resize(2);

  const auto image_sizes = Eigen::Vector2f{1920.f, 1080.f};

  auto line_data = LineRenderer::LineHostData{};

  // Draw a quad.
  line_data.add_line_segment_in_pixel_coordinates(  //
      {0.f, 0.f}, {1920.f, 0.f}, image_sizes, 3.f, 1.f);
  line_data.add_line_segment_in_pixel_coordinates(
      {1920.f, 0.f}, {1920.f, 1080.f}, image_sizes, 3.f, 1.f);
  line_data.add_line_segment_in_pixel_coordinates(
      {1920.f, 1080.f}, {0.f, 1080.f}, image_sizes, 3.f, 1.f);
  line_data.add_line_segment_in_pixel_coordinates(  //
      {0.f, 1080.f}, {0.f, 0.f}, image_sizes, 3.f, 1.f);
  gl_lines[0].set_data(line_data);
  gl_lines[0]._color << .8f, .0f, .0f, .5f;

  line_data.clear();

  // Draw a triangle.
  line_data.add_line_segment_in_pixel_coordinates(  //
      {0.f, 0.f}, {1920.f, 0.f}, image_sizes, 3.f, 1.f);
  line_data.add_line_segment_in_pixel_coordinates(
      {1920.f, 0.f}, {1920.f, 1080.f}, image_sizes, 3.f, 1.f);
  line_data.add_line_segment_in_pixel_coordinates(  //
      {1920.f, 1080.f}, {0.f, 0.f}, image_sizes, 3.f, 1.f);
  gl_lines[1].set_data(line_data);
  gl_lines[1]._color << .0f, .8f, .0f, .5f;
}

auto cleanup_gl_objects() -> void
{
  // Destroy the shaders and quad geometry data.
  auto& image_plane_renderer = ImagePlaneRenderer::instance();
  image_plane_renderer.destroy_gl_objects();

  auto& line_renderer = LineRenderer::instance();
  line_renderer.destroy_gl_objects();

  // Destroy the image textures.
  auto& image_textures = image_plane_renderer._textures;
  for (auto i = 0u; i < image_textures.size(); ++i)
    image_textures[i].destroy();
  image_textures.clear();

  // Clear the list of image textures.
  image_textures.clear();

  // Destroy the lines.
  auto& gl_lines = line_renderer._lines;
  for (auto i = 0u; i < gl_lines.size(); ++i)
    gl_lines[i].destroy();
  gl_lines.clear();
}

int main()
{
  try
  {
    if (!MyGLFW::initialize())
      return EXIT_FAILURE;

#ifndef EMSCRIPTEN
    glewInit();
#endif

    auto& image_plane_renderer = ImagePlaneRenderer::instance();
    image_plane_renderer.initialize();
    initialize_image_textures();

    auto& line_renderer = LineRenderer::instance();
    line_renderer.initialize();
    initialize_lines();

    // Specific rendering options.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // glEnable(GL_DEPTH_TEST);

#ifdef EMSCRIPTEN
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
