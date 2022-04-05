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
#include "MyGLFW.hpp"
#include "LinePainter.hpp"
#include "Scene.hpp"


namespace sara = DO::Sara;

using namespace std;


int main()
{
  try
  {
    if (!MyGLFW::initialize())
      return EXIT_FAILURE;

    auto& painter = LinePainter::instance();
    painter.initialize();

    auto& scene = Scene::instance();
    scene.initialize();

    // Activate the texture 0 once for all.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, scene._texture);

    // Specific rendering options.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // glEnable(GL_DEPTH_TEST);

    // We must use the shader program first before specifying the textures!
    scene._shader_program.use(true);
    {
      // Specify all the matrix uniforms.
      scene._shader_program.set_uniform_matrix4f(
          "transform", scene._transform.matrix().data());
      scene._shader_program.set_uniform_matrix4f("view", scene._view.data());
      scene._shader_program.set_uniform_matrix4f("projection",
                                                 scene._projection.data());

      // Specify the texture uniform.
      const auto tex_location =
          glGetUniformLocation(scene._shader_program, "image");
      if (tex_location == GL_INVALID_VALUE)
        throw std::runtime_error{"Cannot find texture location!"};
      glUniform1i(tex_location, 0);
    }

    // Transfer line data to GPU.
    {
      painter.add_line_segment(Eigen::Vector2f(-0.4, -0.4),
                               Eigen::Vector2f(+0.4, -0.4));
      painter.add_line_segment(Eigen::Vector2f(+0.4, -0.4),
                               Eigen::Vector2f(+0.4, +0.4));
      painter.add_line_segment(Eigen::Vector2f(+0.4, +0.4),
                               Eigen::Vector2f(-0.4, +0.4));
      painter.transfer_line_tesselation_to_gl_buffers();
    }

    // Frame render function.
    auto render_frame = []() {
      glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      auto& scene = Scene::instance();
      scene.render();

      auto& line_painter = LinePainter::instance();
      line_painter.render();

      glfwSwapBuffers(MyGLFW::window);
      glfwPollEvents();
    };

#ifdef EMSCRIPTEN
    emscripten_set_main_loop(render_frame, 0, 1);
#else
    while (!glfwWindowShouldClose(MyGLFW::window))
      render_frame();
#endif

    scene.destroy_opengl_data();
    painter.destroy_opengl_data();

    glfwTerminate();
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
