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

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <Eigen/Geometry>

#include <map>


namespace kgl = DO::Kalpana::GL;
namespace sara = DO::Sara;

using namespace DO::Sara;


inline auto init_glfw_boilerplate()
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

inline auto init_glew_boilerplate()
{
#ifndef __APPLE__
  // Initialize GLEW.
  auto err = glewInit();
  if (err != GLEW_OK)
  {
    std::cerr << sara::format("Error: failed to initialize GLEW: %s",
                        glewGetErrorString(err))
              << std::endl;
  }
#endif
}


auto resize_framebuffer(GLFWwindow*, int width, int height)
{
  // make sure the viewport matches the new window dimensions; note that width
  // and height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}


int main()
{
  init_glfw_boilerplate();

  // Create a window.
  const auto width = 800;
  const auto height = 600;
  auto window =
      glfwCreateWindow(width, height, "Hello Transformations", nullptr, nullptr);
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, resize_framebuffer);

  init_glew_boilerplate();

  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                        {"in_color", 1},   //
                                        {"in_tex_coords", 2},   //
                                        {"out_color", 0}};

  const auto vertex_shader_source = R"shader(
#version 330 core
  layout (location = 0) in vec3 in_coords;
  layout (location = 1) in vec3 in_color;
  layout (location = 2) in vec2 in_tex_coords;

  uniform mat4 transform;

  out vec3 out_color;
  out vec2 out_tex_coords;

  void main()
  {
    gl_Position = transform * vec4(in_coords, 1.0);
    out_color = in_color;
    out_tex_coords = vec2(in_tex_coords.x, in_tex_coords.y);
  }
  )shader";
  auto vertex_shader = kgl::Shader{};
  vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);


  const auto fragment_shader_source = R"shader(
#version 330 core
  in vec3 out_color;
  in vec2 out_tex_coords;
  out vec4 frag_color;

  uniform sampler2D texture0;
  uniform sampler2D texture1;

  void main()
  {
    if (out_tex_coords.x > 0.5)
      frag_color = texture(texture0, out_tex_coords);
    else
      frag_color = texture(texture1, out_tex_coords);
    //frag_color = mix(texture(texture0, out_tex_coords),
    //                 texture(texture1, out_tex_coords), 0.5)
    //           * vec4(out_color, 1.0);
  }
  )shader";
  auto fragment_shader = kgl::Shader{};
  fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                     fragment_shader_source);

  auto shader_program = kgl::ShaderProgram{};
  shader_program.create();
  shader_program.attach(vertex_shader, fragment_shader);

  vertex_shader.destroy();
  fragment_shader.destroy();

  // Encode the vertex data in a tensor.
  auto vertices = Tensor_<float, 2>{{4, 8}};
  vertices.flat_array() << //
    // coords            color              texture coords
     0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  1.0f, 0.0f,  // bottom-right
     0.5f,  0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  1.0f, 1.0f,  // top-right
    -0.5f,  0.5f, 0.0f,  1.0f, 1.0f, 0.0f,  0.0f, 1.0f,  // top-left
    -0.5f, -0.5f, 0.0f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f;  // bottom-left

  auto triangles = Tensor_<unsigned int, 2>{{2, 3}};
  triangles.flat_array() <<
    0, 1, 2,
    2, 3, 0;

  const auto row_bytes = [](const TensorView_<float, 2>& data) {
    return static_cast<GLsizei>(data.size(1) * sizeof(float));
  };
  const auto float_pointer = [](int offset) {
    return reinterpret_cast<void*>(offset * sizeof(float));
  };

  auto vao = kgl::VertexArray{};
  vao.generate();

  // Vertex attributes.
  auto vbo = kgl::Buffer{};
  vbo.generate();

  // Triangles data.
  auto ebo = kgl::Buffer{};
  ebo.generate();

  {
    glBindVertexArray(vao);

    // Copy vertex data.
    vbo.bind_vertex_data(vertices);

    // Copy geometry data.
    ebo.bind_triangles_data(triangles);

    // Map the parameters to the argument position for the vertex shader.
    //
    // Vertex coordinates.
    glVertexAttribPointer(arg_pos["in_coords"], 3 /* 3D points */, GL_FLOAT,
                          GL_FALSE, row_bytes(vertices), float_pointer(0));
    glEnableVertexAttribArray(arg_pos["in_coords"]);

    // Colors.
    glVertexAttribPointer(arg_pos["in_color"], 3 /* 3D colors */, GL_FLOAT,
                          GL_FALSE, row_bytes(vertices), float_pointer(3));
    glEnableVertexAttribArray(arg_pos["in_color"]);

    // Texture coordinates.
    glVertexAttribPointer(arg_pos["in_tex_coords"], 2 /* 3D colors */, GL_FLOAT,
                          GL_FALSE, row_bytes(vertices), float_pointer(6));
    glEnableVertexAttribArray(arg_pos["in_tex_coords"]);
  }

  // Texture data.
  auto texture0 = kgl::Texture2D{};
  {
    // Read the image from the disk.
    auto image = imread<Rgb8>(src_path("../../../../data/ksmall.jpg"));
    // Flip vertically so that the image data matches OpenGL image coordinate
    // system.
    flip_vertically(image);

    // Copy the image to the GPU texture.
    glActiveTexture(GL_TEXTURE0);
    texture0.setup_with_pretty_defaults(image, 0);
  }

  auto texture1 = kgl::Texture2D{};
  {
    // Read the image from the disk.
    auto image = imread<Rgb8>(src_path("../../../../data/sunflowerField.jpg"));
    // Flip vertically so that the image data matches OpenGL image coordinate
    // system.
    flip_vertically(image);

    // Copy the image to the GPU texture.
    glActiveTexture(GL_TEXTURE1);
    texture1.setup_with_pretty_defaults(image, 0);
  }

  shader_program.use(true);
  // Specify that GL_TEXTURE0 is mapped to texture0 in the fragment shader code.
  shader_program.set_uniform_param("texture0", 0);
  // Specify that GL_TEXTURE1 is mapped to texture1 in the fragment shader code.
  shader_program.set_uniform_param("texture1", 1);

  auto timer = Timer{};

  // Display image.
  glfwSwapInterval(1);
  while (!glfwWindowShouldClose(window))
  {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture1);

    auto transform = Transform<float, 3, Eigen::Projective>{};
    transform.setIdentity();
    transform.rotate(AngleAxisf(static_cast<float>(timer.elapsed_ms() / 1000), Vector3f::UnitZ()));
    transform.translate(Vector3f{0.25f, 0.25f, 0.f});

    shader_program.set_uniform_matrix4f("transform", transform.matrix().data());

    // Draw triangles.
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(triangles.size()), GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  vao.destroy();
  vbo.destroy();
  ebo.destroy();

  // Clean up resources.
  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
