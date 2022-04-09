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


auto initialize_camera_parameters() -> void
{
  auto& grid_renderer = MetricGridRenderer::instance();
  auto& intrinsics = grid_renderer._intrinsics;
  // clang-format off
  const auto K = (Eigen::Matrix3f{} <<
    1041.55762f, -2.31719828f, 942.885742f,
            0.f,  1041.53857f, 589.198425f,
            0.f,          0.f,         1.f
  ).finished();
  intrinsics.set_calibration_matrix(K);
  intrinsics.radial_distortion_coefficients <<
       0.442631334f,
      -0.156340882f,
       0;
  intrinsics.tangential_distortion_coefficients <<
      -0.000787709199f,
      -0.000381082471f;
  // clang-format on
  intrinsics.xi = 1.43936455f;

  std::cout << intrinsics.K << std::endl;
}

auto initialize_metric_grid(const std::pair<std::int32_t, std::int32_t>& xrange,
                            const std::pair<std::int32_t, int32_t>& yrange,
                            float square_size_in_meters = 1.f,
                            float line_discretization_step = 0.25f) -> void
{
  auto& grid_renderer = MetricGridRenderer::instance();

  grid_renderer._vertices.clear();
  grid_renderer._triangles.clear();

  const auto& sq_size = square_size_in_meters;
  const auto& s = line_discretization_step;

  // Draw y-level sets.
  for (auto y = static_cast<float>(yrange.first); y <= yrange.second;
       y += sq_size)
  {
    // std::cout << "y = " << y << std::endl;
    for (auto x = static_cast<float>(xrange.first); x < xrange.second; x += s)
    {
      // std::cout << "x = " << x << std::endl;
      const auto a = Eigen::Vector2f(x, y);
      const auto b = Eigen::Vector2f(x + s, y);
      grid_renderer.add_line_segment(a, b, 10.f / 1080, 0.5f / 1080);
    }
  }

  // Draw x-level sets.
  for (auto x = static_cast<float>(xrange.first); x <= xrange.second;
       x += sq_size)
  {
    // std::cout << "x = " << x << std::endl;
    for (auto y = static_cast<float>(yrange.first); y < yrange.second; y += s)
    {
      // std::cout << "y = " << y << std::endl;
      const auto a = Eigen::Vector2f(x, y);
      const auto b = Eigen::Vector2f(x, y + s);
      grid_renderer.add_line_segment(a, b, 10.f / 1080, 0.5f / 1080);
    }
  }
}

auto render_frame() -> void
{
  // TODO: sort the projective transformation later and so on.
  glViewport(0, 0, MyGLFW::width, MyGLFW::height);

  glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  auto& scene = Scene::instance();
  scene.render();

  auto& grid_renderer = MetricGridRenderer::instance();
  grid_renderer.render();

  auto& line_renderer = LinePainter::instance();
  line_renderer.render();

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

    auto& line_renderer = LinePainter::instance();
    line_renderer.initialize();
    line_renderer.add_line_segment_in_pixel_coordinates(
        {0.f, 0.f}, {1920.f, 0.f}, 3.f, 1.f);
    line_renderer.add_line_segment_in_pixel_coordinates(
        {1920.f, 0.f}, {1920.f, 1080.f}, 3.f, 1.f);
    line_renderer.transfer_line_tesselation_to_gl_buffers();

    auto& grid_renderer = MetricGridRenderer::instance();
    grid_renderer.initialize();
    initialize_camera_parameters();
    initialize_metric_grid({5, 100}, {-100, 100});
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
