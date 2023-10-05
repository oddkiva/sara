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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>

#include <DO/Kalpana/Math/Projection.hpp>
#include <array>

#ifdef _WIN32
#  include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <Eigen/Geometry>

#include <filesystem>
#include <map>


using namespace DO::Sara;


namespace fs = std::filesystem;
namespace k = DO::Kalpana;
namespace kgl = k::GL;
namespace sara = DO::Sara;


auto resize_framebuffer(GLFWwindow*, int width, int height)
{
  // make sure the viewport matches the new window dimensions; note that width
  // and height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}

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
  const auto err = glewInit();
  if (err != GLEW_OK)
  {
    std::cerr << sara::format("Error: failed to initialize GLEW: %s",
                              glewGetErrorString(err))
              << std::endl;
  }
#endif
}


auto make_cube()
{
  auto cube = Tensor_<float, 2>{6 * 6, 5};
  // clang-format off
  cube.flat_array() <<
    -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
     0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
     0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
    -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

    -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
     0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
     0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
     0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
     0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
     0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
     0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f;
  // clang-format on
  return cube;
}


int main(int, char** argv)
{
  // Iniialize the GLFW application.
  init_glfw_boilerplate();

  // Create a window.
  const auto width = 800;
  const auto height = 600;
  auto window = glfwCreateWindow(width, height,               //
                                 "Hello Coordinate Systems",  //
                                 nullptr, nullptr);
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, resize_framebuffer);

  // Import OpenGL API dynamically.
  init_glew_boilerplate();

  static const auto arg_pos = std::map<std::string, int>{
      {"in_coords", 0},      //
      {"in_tex_coords", 1},  //
      {"out_color", 0}       //
  };

  static constexpr auto vertex_shader_source = R"shader(
#version 330 core
  layout (location = 0) in vec3 in_coords;
  layout (location = 1) in vec2 in_tex_coords;

  uniform mat4 transform;
  uniform mat4 view;
  uniform mat4 projection;

  out vec3 out_color;
  out vec2 out_tex_coords;

  void main()
  {
    gl_Position = projection * view * transform * vec4(in_coords, 1.0);
    gl_PointSize = 5.f;
    out_tex_coords = vec2(in_tex_coords.x, in_tex_coords.y);
  }
  )shader";
  auto vertex_shader = kgl::Shader{};
  vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);


  static constexpr auto fragment_shader_source = R"shader(
#version 330 core
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
  auto vertices = make_cube();

  // clang-format off
  static const auto cube_positions = std::array<Vector3f, 10>{
      Vector3f( 0.0f,  0.0f,  0.0f), Vector3f( 2.0f,  5.0f, -15.0f),
      Vector3f(-1.5f, -2.2f, -2.5f), Vector3f(-3.8f, -2.0f, -12.3f),
      Vector3f( 2.4f, -0.4f, -3.5f), Vector3f(-1.7f,  3.0f, -7.5f),
      Vector3f( 1.3f, -2.0f, -2.5f), Vector3f( 1.5f,  2.0f, -2.5f),
      Vector3f( 1.5f,  0.2f, -1.5f), Vector3f(-1.3f,  1.0f, -1.5f)
  };
  // clang-format on

  static const auto row_bytes = [](const TensorView_<float, 2>& data) {
    return static_cast<GLsizei>(data.size(1) * sizeof(float));
  };
  static const auto float_pointer = [](const int offset) {
    return reinterpret_cast<void*>(offset * sizeof(float));
  };

  auto vao = kgl::VertexArray{};
  vao.generate();

  // Vertex attributes.
  auto vbo = kgl::Buffer{};
  vbo.generate();
  {
    glBindVertexArray(vao);

    // Copy vertex data.
    vbo.bind_vertex_data(vertices);

    // Map the parameters to the argument position for the vertex shader.
    //
    // Vertex coordinates.
    glVertexAttribPointer(arg_pos.at("in_coords"), 3 /* 3D points */, GL_FLOAT,
                          GL_FALSE, row_bytes(vertices), float_pointer(0));
    glEnableVertexAttribArray(arg_pos.at("in_coords"));

    // Texture coordinates.
    glVertexAttribPointer(arg_pos.at("in_tex_coords"), 2 /* texture coords */,
                          GL_FLOAT, GL_FALSE, row_bytes(vertices),
                          float_pointer(3));
    glEnableVertexAttribArray(arg_pos.at("in_tex_coords"));
  }

  // Texture data.
  static const auto program_dir_path =
      fs::absolute(fs::path(argv[0])).parent_path();
  auto texture0 = kgl::Texture2D{};
  {
    // Read the image from the disk.
    auto image = imread<Rgb8>(                               //
        (program_dir_path / "data" / "ksmall.jpg").string()  //
    );
    // Flip vertically so that the image data matches OpenGL image coordinate
    // system.
    flip_vertically(image);

    // Copy the image to the GPU texture.
    glActiveTexture(GL_TEXTURE0);
    texture0.setup_with_pretty_defaults(image);
  }

  auto texture1 = kgl::Texture2D{};
  {
    // Read the image from the disk.
    auto image = imread<Rgb8>(                                       //
        (program_dir_path / "data" / "sunflowerField.jpg").string()  //
    );
    // Flip vertically so that the image data matches OpenGL image coordinate
    // system.
    flip_vertically(image);

    // Copy the image to the GPU texture.
    glActiveTexture(GL_TEXTURE1);
    texture1.setup_with_pretty_defaults(image);
  }

  shader_program.use(true);
  // Specify that GL_TEXTURE0 is mapped to texture0 in the fragment shader code.
  shader_program.set_uniform_param("texture0", 0);
  // Specify that GL_TEXTURE1 is mapped to texture1 in the fragment shader code.
  shader_program.set_uniform_param("texture1", 1);

  const auto view_uniform = shader_program.get_uniform_location("view");
  const auto proj_uniform = shader_program.get_uniform_location("projection");
  const auto tsfm_uniform = shader_program.get_uniform_location("transform");

  auto timer = Timer{};

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_PROGRAM_POINT_SIZE);

  // You need this for 3D objects!
  glEnable(GL_DEPTH_TEST);
  // Backgoun color
  glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
  glBindTexture(GL_TEXTURE_2D, texture0);
  glBindTexture(GL_TEXTURE_2D, texture1);

  // Display image.
  glfwSwapInterval(1);
  while (!glfwWindowShouldClose(window))
  {
    // Important.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    auto view = Transform<float, 3, Eigen::Projective>{};
    view.setIdentity();
    view.translate(Vector3f{0.f, 0.f, -10.f});
    shader_program.set_uniform_matrix4f(view_uniform, view.matrix().data());

    const Matrix4f projection = k::perspective(45.f, 800.f / 600.f, .1f, 100.f);
    shader_program.set_uniform_matrix4f(proj_uniform, projection.data());

    // Draw triangles.
    glBindVertexArray(vao);
    for (int i = 0; i < 10; ++i)
    {
      auto transform = Transform<float, 3, Eigen::Projective>{};
      transform.setIdentity();
      transform.translate(cube_positions[i]);
      transform.rotate(
          AngleAxisf(static_cast<float>(std::pow(1.2, (i + 1) * 5) *
                                        timer.elapsed_ms() / 10000),
                     Vector3f{0.5f, 1.0f, 0.0f}.normalized()));
      shader_program.set_uniform_matrix4f(tsfm_uniform,
                                          transform.matrix().data());

      glDrawArrays(GL_TRIANGLES, 0, vertices.size(0));
    }


    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // Destroy OpenGL resources.
  vao.destroy();
  vbo.destroy();

  // Clean up GLFW.
  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
