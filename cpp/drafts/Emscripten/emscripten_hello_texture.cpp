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
#include <DO/Sara/Core/StringFormat.hpp>

#include <DO/Sara/ImageIO.hpp>

#include <iostream>
#include <map>
#include <memory>

#ifdef EMSCRIPTEN
#  include <emscripten/emscripten.h>
#  define GLFW_INCLUDE_ES3
#endif

#include "Geometry.hpp"
#include "LinePainter.hpp"
#include "MetricGridRenderer.hpp"
#include "Scene.hpp"

#include "MyGLFW.hpp"


namespace sara = DO::Sara;

using namespace std;


// Frame render function.
auto render_frame() -> void
{
  glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  auto& scene = Scene::instance();
  scene.render();

  static float t = 0;
  static constexpr auto dt = 1.f / (180.f) * float(M_PI);

  static constexpr auto thickness = 10.f / 1080.f; // 10 pixels in normalized coordinates...
  static constexpr auto antialias_radius = 0.5f / 1080.f;

  // Dummy animation, just to show the possibilities.
  //
  // Transfer line data to GPU.
  auto& line_painter = LinePainter::instance();
  {
    line_painter._vertices.clear();
    line_painter._triangles.clear();
    line_painter.add_line_segment(Eigen::Vector2f(-0.4 + std::cos(t), -0.4),
                                  Eigen::Vector2f(+0.4, -0.4),  //
                                  thickness, antialias_radius);
    line_painter.add_line_segment(Eigen::Vector2f(+0.4, -0.4 + std::sin(t)),
                                  Eigen::Vector2f(+0.4, +0.4),  //
                                  thickness, antialias_radius);
    line_painter.add_line_segment(Eigen::Vector2f(+0.4, +0.4),
                                  Eigen::Vector2f(-0.4, +0.4),  //
                                  thickness, antialias_radius);
    line_painter.transfer_line_tesselation_to_gl_buffers();

    t += dt;
    if (t > 2 * M_PI)
      t = 0.f;
  }

  line_painter.render();

  auto& grid_renderer = MetricGridRenderer::instance();
  grid_renderer.render();

  glfwSwapBuffers(MyGLFW::window);
  glfwPollEvents();
}

int main()
{
  try
  {
    if (!MyGLFW::initialize())
      return EXIT_FAILURE;

    auto& scene = Scene::instance();
    const auto image = sara::imread<sara::Rgb8>("assets/image.png");
    scene.initialize(image);

    auto& painter = LinePainter::instance();
    painter.initialize();

    auto& grid_renderer = MetricGridRenderer::instance();
    grid_renderer.initialize();
    grid_renderer.add_line_segment({-0.4f, -0.4f}, {+0.4f, +0.4f});
    grid_renderer.transfer_line_tesselation_to_gl_buffers();

    // Activate the texture 0 once for all.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, scene._texture);

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

    scene.destroy_opengl_data();
    painter.destroy_opengl_data();
    grid_renderer.destroy_opengl_data();

    glfwTerminate();
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
