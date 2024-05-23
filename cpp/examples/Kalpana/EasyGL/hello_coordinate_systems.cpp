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
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>

#include <DO/Kalpana/Math/Projection.hpp>

#include <fmt/format.h>

#ifdef _WIN32
#  include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <Eigen/Geometry>

#include <array>
#include <filesystem>
#include <map>


using namespace DO::Sara;


namespace fs = std::filesystem;
namespace k = DO::Kalpana;
namespace kgl = k::GL;
namespace sara = DO::Sara;

using Eigen::Vector3f;


struct ShaderProgramBuilder
{
  const std::map<std::string, int> arg_pos = {
      {"in_coords", 0},      //
      {"in_tex_coords", 1},  //
      {"out_color", 0}       //
  };

  auto row_bytes(const sara::TensorView_<float, 2>& data) const -> GLsizei
  {
    return static_cast<GLsizei>(data.size(1) * sizeof(float));
  }

  auto float_pointer(const int offset) const -> void*
  {
    return reinterpret_cast<void*>(offset * sizeof(float));
  }

  auto build_shader_program() const -> kgl::ShaderProgram
  {
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

    return shader_program;
  }
};


class App
{
public:
  App(const fs::path& program_dir_path,  //
      const Eigen::Vector2i& sizes, const std::string& title)
    : _program_dir_path{program_dir_path}
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
    _shader_program.use(true);
    // Specify that GL_TEXTURE0 is mapped to texture0 in the fragment shader
    // code.
    _shader_program.set_uniform_param("texture0", 0);
    // Specify that GL_TEXTURE1 is mapped to texture1 in the fragment shader
    // code.
    _shader_program.set_uniform_param("texture1", 1);

    const auto view_uniform = _shader_program.get_uniform_location("view");
    const auto proj_uniform =
        _shader_program.get_uniform_location("projection");
    const auto tsfm_uniform = _shader_program.get_uniform_location("transform");

    auto timer = Timer{};

    while (!glfwWindowShouldClose(_window))
    {
      // Important.
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      auto view = Eigen::Transform<float, 3, Eigen::Projective>{};
      view.setIdentity();
      view.translate(Vector3f{0.f, 0.f, -10.f});
      _shader_program.set_uniform_matrix4f(view_uniform, view.matrix().data());

      const auto projection = k::perspective(45.f, 800.f / 600.f, .1f, 100.f);
      _shader_program.set_uniform_matrix4f(proj_uniform, projection.data());

      // Draw triangles.
      glBindVertexArray(_vao);
      for (int i = 0; i < 10; ++i)
      {
        auto transform = Transform<float, 3, Eigen::Projective>{};
        transform.setIdentity();
        transform.translate(_cube_positions[i]);
        const auto angle = static_cast<float>(std::pow(1.2, (i + 1) * 5) *
                                              timer.elapsed_ms() / 10000);
        const Vector3f axis = Vector3f{0.5f, 1.0f, 0.0f}.normalized();
        transform.rotate(Eigen::AngleAxisf{angle, axis});
        _shader_program.set_uniform_matrix4f(tsfm_uniform,
                                             transform.matrix().data());

        glDrawArrays(GL_TRIANGLES, 0, _num_vertices);
      }


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
    const auto vertices = make_cube();
    _num_vertices = vertices.size();

    _vao.generate();
    _vbo.generate();

    glBindVertexArray(_vao);

    const auto shader_program_builder = ShaderProgramBuilder{};
    const auto& shader_arg_pos = shader_program_builder.arg_pos;

    // Copy vertex data.
    _vbo.bind_vertex_data(vertices);

    // Map the parameters to the argument position for the vertex shader.
    //
    // Vertex coordinates.
    glVertexAttribPointer(shader_arg_pos.at("in_coords"), 3 /* 3D points */,
                          GL_FLOAT, GL_FALSE,
                          shader_program_builder.row_bytes(vertices),
                          shader_program_builder.float_pointer(0));
    glEnableVertexAttribArray(shader_arg_pos.at("in_coords"));

    // Texture coordinates.
    glVertexAttribPointer(shader_arg_pos.at("in_tex_coords"),
                          2 /* texture coords */, GL_FLOAT, GL_FALSE,
                          shader_program_builder.row_bytes(vertices),
                          shader_program_builder.float_pointer(3));
    glEnableVertexAttribArray(shader_arg_pos.at("in_tex_coords"));

    // Texture data.
    {
      // Read the image from the disk.
      auto image = sara::imread<sara::Rgb8>(                    //
          (_program_dir_path / "data" / "ksmall.jpg").string()  //
      );
      // Flip vertically so that the image data matches OpenGL image coordinate
      // system.
      flip_vertically(image);

      // Copy the image to the GPU texture.
      glActiveTexture(GL_TEXTURE0);
      _textures[0].setup_with_pretty_defaults(image);
    }

    {
      // Read the image from the disk.
      auto image = sara::imread<sara::Rgb8>(                            //
          (_program_dir_path / "data" / "sunflowerField.jpg").string()  //
      );
      // Flip vertically so that the image data matches OpenGL image coordinate
      // system.
      flip_vertically(image);

      // Copy the image to the GPU texture.
      glActiveTexture(GL_TEXTURE1);
      _textures[1].setup_with_pretty_defaults(image);
    }
  }

  auto init_render_settings() -> void
  {
    // You need this for 3D objects!
    glEnable(GL_DEPTH_TEST);

    // Backgound color
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    // Textures.
    glBindTexture(GL_TEXTURE_2D, _textures[0]);
    glBindTexture(GL_TEXTURE_2D, _textures[1]);

    // Animation settings.
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
      throw std::runtime_error{
          fmt::format("Error: failed to initialize GLEW: {}",
                      reinterpret_cast<const char*>(glewGetErrorString(err)))};
#endif
  }

  static auto make_cube() -> sara::Tensor_<float, 2>
  {
    auto cube = sara::Tensor_<float, 2>{6 * 6, 5};
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


private:
  static bool _glfw_initialized;
  fs::path _program_dir_path;

  GLFWwindow* _window = nullptr;

  // The graphics pipeline.
  kgl::ShaderProgram _shader_program;

  // clang-format off
  std::array<Vector3f, 10> _cube_positions = {
    Vector3f( 0.0f,  0.0f,  0.0f), Vector3f( 2.0f,  5.0f, -15.0f),
    Vector3f(-1.5f, -2.2f, -2.5f), Vector3f(-3.8f, -2.0f, -12.3f),
    Vector3f( 2.4f, -0.4f, -3.5f), Vector3f(-1.7f,  3.0f, -7.5f),
    Vector3f( 1.3f, -2.0f, -2.5f), Vector3f( 1.5f,  2.0f, -2.5f),
    Vector3f( 1.5f,  0.2f, -1.5f), Vector3f(-1.3f,  1.0f, -1.5f)
  };
  // clang-format on

  // Geometry data on OpenGL side.
  GLsizei _num_vertices;
  kgl::VertexArray _vao;
  kgl::Buffer _vbo;
  kgl::Buffer _ebo;

  std::array<kgl::Texture2D, 2> _textures;
};

bool App::_glfw_initialized = false;


auto main(int, char** argv) -> int
{
  try
  {
    static constexpr auto width = 800;
    static constexpr auto height = 600;
    static constexpr auto title = "Hello Transformations";
    static const auto program_dir_path =
        fs::absolute(fs::path(argv[0])).parent_path();

    auto app = App{program_dir_path, {width, height}, title};
    app.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
