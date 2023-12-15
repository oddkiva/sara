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
#include <DO/Sara/Defines.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>

#if defined(_WIN32)
#  include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <filesystem>
#include <map>


namespace sara = DO::Sara;
namespace kgl = DO::Kalpana::GL;
namespace fs = std::filesystem;


struct ShaderProgramBuilder
{
  const std::map<std::string, int> arg_pos = {{"in_coords", 0},      //
                                              {"in_color", 1},       //
                                              {"in_tex_coords", 2},  //
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
  layout (location = 2) in vec2 in_tex_coords;

  out vec3 out_color;
  out vec2 out_tex_coords;

  void main()
  {
    gl_Position = vec4(in_coords, 1.0);
    gl_PointSize = 200.0;
    out_color = in_color;
    out_tex_coords = vec2(in_tex_coords.x, in_tex_coords.y);
  }
    )shader";
    auto vertex_shader = kgl::Shader{};
    vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);


    static constexpr auto fragment_shader_source = R"shader(
#version 330 core
  in vec3 out_color;
  in vec2 out_tex_coords;
  out vec4 frag_color;

  uniform sampler2D texture1;

  void main()
  {
    frag_color = texture(texture1, out_tex_coords) * vec4(out_color, 1.0);
  }
    )shader";
    auto fragment_shader = kgl::Shader{};
    fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                       fragment_shader_source);

    auto shader_program = kgl::ShaderProgram{};
    shader_program.create();
    shader_program.attach(vertex_shader, fragment_shader);

    shader_program.detach();
    vertex_shader.destroy();
    fragment_shader.destroy();

    return shader_program;
  }
};


class App
{
public:
  App(const fs::path& program_dirpath, const Eigen::Vector2i& sizes,
      const std::string& title)
    : _program_dirpath{program_dirpath}
  {
    // Init GLFW.
    init_glfw();

    // Create a GLFW window.
    _window = glfwCreateWindow(sizes.x(), sizes.y(),  //
                               title.c_str(),         //
                               nullptr, nullptr);

    // Prepare OpenGL first before any OpenGL calls.
    init_opengl();

    init_shader_program();
    init_data_on_opengl();
    init_render_settings();
  }

  ~App()
  {
    _vao.destroy();
    _vbo.destroy();
    _ebo.destroy();
    _texture.destroy();

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
    _shader_program.use();
    while (!glfwWindowShouldClose(_window))
    {
      glClear(GL_COLOR_BUFFER_BIT);

      // Draw triangles.
      glBindVertexArray(_vao);
      glDrawElements(GL_TRIANGLES, _num_triangle_indices, GL_UNSIGNED_INT, 0);

      glfwSwapBuffers(_window);
      glfwPollEvents();
    }
  }

private:
  auto init_opengl() -> void
  {
    // GLFW context...
    glfwMakeContextCurrent(_window);

    // Init OpenGL extensions.
    init_glew();
  }

  auto init_shader_program() -> void
  {
    const auto shader_program_builder = ShaderProgramBuilder{};
    _shader_program = shader_program_builder.build_shader_program();
  }

  auto init_data_on_opengl() -> void
  {
    // Encode the vertex data in a tensor.
    auto vertices = sara::Tensor_<float, 2>{{4, 8}};
    // clang-format off
    vertices.flat_array() <<
    // coords              color              texture coords
       0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  1.0f, 0.0f,  // bottom-right
       0.5f,  0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  1.0f, 1.0f,  // top-right
      -0.5f,  0.5f, 0.0f,  1.0f, 1.0f, 0.0f,  0.0f, 1.0f,  // top-left
      -0.5f, -0.5f, 0.0f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f;  // bottom-left
    // clang-format on

    auto triangles = sara::Tensor_<unsigned int, 2>{{2, 3}};
    // clang-format off
    triangles.flat_array() <<
      0, 1, 2,
      2, 3, 0;
    // clang-format on
    _num_triangle_indices = triangles.size();

    _vao.generate();

    // Vertex attributes.
    _vbo.generate();
    // Triangles data.
    _ebo.generate();

    glBindVertexArray(_vao);

    // Copy vertex data.
    _vbo.bind_vertex_data(vertices);

    // Copy geometry data.
    _ebo.bind_triangles_data(triangles);

    const auto shader_program_builder = ShaderProgramBuilder{};
    const auto& shader_arg_pos = shader_program_builder.arg_pos;

    // Map the parameters to the argument position for the vertex shader.
    //
    // Vertex coordinates.
    glVertexAttribPointer(shader_arg_pos.at("in_coords"), 3 /* 3D points */,
                          GL_FLOAT, GL_FALSE,
                          shader_program_builder.row_bytes(vertices),
                          shader_program_builder.float_pointer(0));
    glEnableVertexAttribArray(shader_arg_pos.at("in_coords"));

    // Colors.
    glVertexAttribPointer(shader_arg_pos.at("in_color"), 3 /* 3D colors */,
                          GL_FLOAT, GL_FALSE,
                          shader_program_builder.row_bytes(vertices),
                          shader_program_builder.float_pointer(3));
    glEnableVertexAttribArray(shader_arg_pos.at("in_color"));

    // Texture coordinates.
    glVertexAttribPointer(shader_arg_pos.at("in_tex_coords"),
                          2 /* 2D texture coords */, GL_FLOAT, GL_FALSE,
                          shader_program_builder.row_bytes(vertices),
                          shader_program_builder.float_pointer(6));
    glEnableVertexAttribArray(shader_arg_pos.at("in_tex_coords"));

    // Read the image from the disk.
    auto image = sara::imread<sara::Rgb8>(
        (_program_dirpath / "data" / "ksmall.jpg").string());
    // Flip vertically so that the image data matches OpenGL image coordinate
    // system.
    flip_vertically(image);

    // Copy the image to the GPU texture.
    _texture.setup_with_pretty_defaults(image, 0);
  }

  auto init_render_settings() -> void
  {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    // Activate the texture 0 once for all.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _texture);

    glfwSwapInterval(1);
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
  const fs::path _program_dirpath;

  GLFWwindow* _window = nullptr;

  //! @brief The graphics pipeline.
  kgl::ShaderProgram _shader_program;

  //! @brief Geometry data on OpenGL side.
  //! @{
  GLsizei _num_triangle_indices;
  kgl::VertexArray _vao;
  kgl::Buffer _vbo;
  kgl::Buffer _ebo;
  //! @}

  //! @brief Image texture.
  kgl::Texture2D _texture;
};

bool App::_glfw_initialized = false;


auto main(const int, char** const argv) -> int
{
  try
  {
    static constexpr auto width = 800;
    static constexpr auto height = 600;
    static constexpr auto title = "Hello Texture";

    const auto program_dirpath = fs::absolute(fs::path(argv[0]).parent_path());
    auto app = App{program_dirpath, {width, height}, title};
    app.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
