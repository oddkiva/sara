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
#include <DO/Sara/Core/StringFormat.hpp>

#if defined(_WIN32)
#  include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <map>


namespace kgl = DO::Kalpana::GL;
namespace sara = DO::Sara;


struct ShaderProgramBuilder
{
  const std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                              {"in_color", 1},   //
                                              {"out_color", 0}};


  auto row_bytes(const sara::TensorView_<float, 2>& data) const -> GLsizei
  {
    return static_cast<GLsizei>(data.size(1) * sizeof(float));
  }

  auto float_pointer(const int offset) const -> void*
  {
    return reinterpret_cast<void*>(offset * sizeof(float));
  };

  auto build_shader_program() const -> kgl::ShaderProgram
  {
    static constexpr auto vertex_shader_source = R"shader(
#version 330 core
  layout (location = 0) in vec3 in_coords;
  layout (location = 1) in vec3 in_color;

  out vec3 out_color;

  void main()
  {
    gl_Position = vec4(in_coords, 1.0);
    gl_PointSize = 200.0;
    out_color = in_color;
  }
  )shader";
    auto vertex_shader = kgl::Shader{};
    vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);

    static constexpr auto fragment_shader_source = R"shader(
#version 330 core
  in vec3 out_color;
  out vec4 frag_color;

  void main()
  {
    frag_color = vec4(out_color, 1.0);
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

    return shader_program;
  }
};


class App
{
public:
  App(const Eigen::Vector2i& sizes, const std::string& title)
  {
    // Init GLFW.
    init_glfw();

    // Create a GLFW window.
    _window = glfwCreateWindow(sizes.x(), sizes.y(),  //
                               title.c_str(),         //
                               nullptr, nullptr);
    glfwMakeContextCurrent(_window);
    glfwSetFramebufferSizeCallback(_window, resize_framebuffer);

    // Prepare OpenGL first before any OpenGL calls.
    init_opengl();

    init_shader_program();
    init_data_on_opengl();
    init_render_settings();
  }

  ~App()
  {
    _ebo.destroy();
    _vbo.destroy();
    _vao.destroy();
    _shader_program.clear();

    // Clean up resources.
    if (_window)
      glfwDestroyWindow(_window);

    // Kill the GLFW app.
    if (_glfw_initialized)
      glfwTerminate();
  }

  auto run() -> void
  {
    // Activate the shader program once and for all.
    _shader_program.use(true);

    while (!glfwWindowShouldClose(_window))
    {
      glClear(GL_COLOR_BUFFER_BIT);

      // Draw triangles
      glBindVertexArray(_vao);
      glDrawElements(GL_TRIANGLES, _num_triangle_indices, GL_UNSIGNED_INT, 0);

      glfwSwapBuffers(_window);
      glfwPollEvents();
    }
  }

private:
  auto init_opengl() -> void
  {
    // Set the GLFW context in order to use OpenGL.
    glfwMakeContextCurrent(_window);

    // Now init OpenGL.
    init_glew();
  }

  auto init_shader_program() -> void
  {
    const auto shader_program_builder = ShaderProgramBuilder{};
    _shader_program = shader_program_builder.build_shader_program();
  }

  auto init_data_on_opengl() -> void
  {
    // clang-format off
    auto vertices = sara::Tensor_<float, 2>{{4, 6}};
    vertices.flat_array() << //
      // coords            color
       0.5f,  0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  // top-right
       0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  // bottom-right
      -0.5f, -0.5f, 0.0f,  0.0f, 0.0f, 1.0f,  // bottom-left
      -0.5f,  0.5f, 0.0f,  1.0f, 1.0f, 0.0f;  // top-left

    auto triangles = sara::Tensor_<std::uint32_t, 2>{{2, 3}};
    triangles.flat_array() <<
      0, 1, 2,
      2, 3, 0;
    // clang-format on
    _num_triangle_indices = triangles.size();

    _vao.generate();
    _vbo.generate();
    _ebo.generate();

    const auto shader_program_builder = ShaderProgramBuilder{};
    const auto& shader_arg_pos = shader_program_builder.arg_pos;

    // Specify the vertex attributes here.
    glBindVertexArray(_vao);

    // Copy the vertex data into the GPU buffer object.
    _vbo.bind_vertex_data(vertices);

    // Copy the triangles data into the GPU buffer object.
    _ebo.bind_triangles_data(triangles);

    // Specify that the vertex shader param 0 corresponds to the first 3 float
    // data of the buffer object.
    glVertexAttribPointer(shader_arg_pos.at("in_coords"), 3 /* 3D points */,
                          GL_FLOAT, GL_FALSE,
                          shader_program_builder.row_bytes(vertices),
                          shader_program_builder.float_pointer(0));
    glEnableVertexAttribArray(shader_arg_pos.at("in_coords"));

    // Specify that the vertex shader param 1 corresponds to the first 3 float
    // data of the buffer object.
    glVertexAttribPointer(shader_arg_pos.at("in_color"), 3 /* 3D colors */,
                          GL_FLOAT, GL_FALSE,
                          shader_program_builder.row_bytes(vertices),
                          shader_program_builder.float_pointer(3));
    glEnableVertexAttribArray(shader_arg_pos.at("in_color"));

    // Unbind the vbo to protect its data.
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
  }


  auto init_render_settings() -> void
  {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    glBindTexture(GL_TEXTURE_2D, _textures[0]);
    glBindTexture(GL_TEXTURE_2D, _textures[1]);

    glfwSwapInterval(1);
  }

private:
  static auto resize_framebuffer(GLFWwindow*, int width, int height) -> void
  {
    // make sure the viewport matches the new window dimensions; note that width
    // and height will be significantly larger than specified on retina
    // displays.
    glViewport(0, 0, width, height);
  }

private: /* convenience free functions*/
  static auto init_glfw() -> void
  {
    if (_glfw_initialized)
      throw std::runtime_error{
          "Error: cannot instantiate more than one GLFW Application!"};

    // Initialize the windows manager.
    _glfw_initialized = glfwInit();
    if (!_glfw_initialized)
      throw std::runtime_error{"Error: failed to initialize GLFW!"};

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
  }

  static auto init_glew() -> void
  {
#if !defined(__APPLE__)
    // Initialize GLEW.
    const auto err = glewInit();
    if (err != GLEW_OK)
      throw std::runtime_error{sara::format(
          "Error: failed to initialize GLEW: %s", glewGetErrorString(err))};
#endif
  }

private:
  static bool _glfw_initialized;

  GLFWwindow* _window = nullptr;

  // The graphics pipeline.
  kgl::ShaderProgram _shader_program;

  // Geometry data on OpenGL side.
  GLsizei _num_triangle_indices;
  kgl::VertexArray _vao;
  kgl::Buffer _vbo;
  kgl::Buffer _ebo;

  std::array<kgl::Texture2D, 2> _textures;
};

bool App::_glfw_initialized = false;

auto main() -> int
{
  try
  {
    static constexpr auto width = 800;
    static constexpr auto height = 600;
    static constexpr auto title = "Hello Transformations";

    auto app = App{{width, height}, title};
    app.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
