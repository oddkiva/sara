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

#if defined(_WIN32)
#  include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <Eigen/Geometry>

#include <map>


namespace k = DO::Kalpana;
namespace kgl = DO::Kalpana::GL;
namespace sara = DO::Sara;


auto read_point_cloud(const std::string& h5_filepath) -> sara::Tensor_<float, 2>
{
  auto h5_file = sara::H5File{h5_filepath, H5F_ACC_RDONLY};

  auto coords = Eigen::MatrixXd{};
  h5_file.read_dataset("points", coords);
  coords.transposeInPlace();
  coords.col(1) *= -1;
  coords.col(2) *= -1;
  SARA_DEBUG << "Read coords OK" << std::endl;
  SARA_DEBUG << "coords =\n" << coords.topRows(20) << std::endl;

  auto colors = sara::Tensor_<double, 2>{};
  h5_file.read_dataset("colors", colors);
  SARA_DEBUG << "Read colors OK" << std::endl;
  SARA_DEBUG << "colors =\n" << colors.matrix().topRows(20) << std::endl;

  // Concatenate the data.
  auto vertex_data = sara::Tensor_<double, 2>{{coords.rows(), 6}};
  vertex_data.matrix() << coords.matrix(), colors.matrix();

  return vertex_data.cast<float>();
}

auto make_point_cloud()
{
  // Encode the vertex data in a tensor.
#if defined(__APPLE__)
  const auto vertex_data = read_point_cloud("/Users/oddkiva/Desktop/geometry.h5");
#else
  const auto vertex_data = read_point_cloud("/home/david/Desktop/geometry.h5");
#endif
  SARA_DEBUG << "vertices =\n" << vertex_data.matrix() << std::endl;
  return vertex_data;
}


struct ShaderProgramBuilder
{
  const std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                              {"in_color", 1}};


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

  uniform mat4 transform;
  uniform mat4 view;
  uniform mat4 projection;

  out vec3 out_color;

  void main()
  {
    gl_Position = projection * view * transform * vec4(in_coords, 1.0);
    gl_PointSize = 5.0;
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
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    if (dot(circCoord, circCoord) > 1.0)
        discard;

    float dist = length(gl_PointCoord - vec2(0.5));
    float alpha = 1.0 - smoothstep(0.2, 0.5, dist);

    frag_color = vec4(out_color, alpha);
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
    glfwSetFramebufferSizeCallback(_window, resize_framebuffer);

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

    const auto view_uniform = _shader_program.get_uniform_location("view");
    const auto proj_uniform =
        _shader_program.get_uniform_location("projection");
    const auto tsfm_uniform = _shader_program.get_uniform_location("transform");


    auto timer = sara::Timer{};
    while (!glfwWindowShouldClose(_window))
    {
      // Important.
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      // Transform matrix.
      auto transform = Eigen::Transform<float, 3, Eigen::Projective>{};
      transform.setIdentity();
      const auto angle =
          static_cast<float>(std::pow(1.5, 5) * timer.elapsed_ms() / 10000);
      const Eigen::Vector3f axis =
          Eigen::Vector3f{0.5f, 1.0f, 0.0f}.normalized();
      transform.rotate(Eigen::AngleAxisf{angle, axis});
      _shader_program.set_uniform_matrix4f(tsfm_uniform,
                                           transform.matrix().data());

      // View matrix.
      auto view = Eigen::Transform<float, 3, Eigen::Projective>{};
      view.setIdentity();
      view.translate(Eigen::Vector3f{0.f, 0.f, -100.f});
      _shader_program.set_uniform_matrix4f(view_uniform, view.matrix().data());

      // Projection matrix.
      const auto projection = k::perspective(45.f, 800.f / 600.f, .1f, 1000.f);
      _shader_program.set_uniform_matrix4f(proj_uniform, projection.data());

      // Draw triangles.
      glBindVertexArray(_vao);
      glDrawArrays(GL_POINTS, 0, _num_vertices);

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
    auto vertices = make_point_cloud();
    _num_vertices = vertices.size(0);

    _vao.generate();
    _vbo.generate();

    glBindVertexArray(_vao);

    // Copy vertex data.
    _vbo.bind_vertex_data(vertices);

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

    // Texture coordinates.
    glVertexAttribPointer(shader_arg_pos.at("in_color"), 3 /* RGB_COLOR */,
                          GL_FLOAT, GL_FALSE,
                          shader_program_builder.row_bytes(vertices),
                          shader_program_builder.float_pointer(3));
    glEnableVertexAttribArray(shader_arg_pos.at("in_color"));
  }

  auto init_render_settings() -> void
  {
    // Setup options for point cloud rendering.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    // Default background color.
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    // You absolutely need this for 3D objects!
    glEnable(GL_DEPTH_TEST);
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
  static auto resize_framebuffer(GLFWwindow*, int width, int height) -> void
  {
    // make sure the viewport matches the new window dimensions; note that width
    // and height will be significantly larger than specified on retina
    // displays.
    glViewport(0, 0, width, height);
  }

private:
  static bool _glfw_initialized;

  GLFWwindow* _window = nullptr;

  //! @brief The graphics pipeline.
  kgl::ShaderProgram _shader_program;

  //! @brief Geometry data on OpenGL side.
  //! @{
  GLsizei _num_vertices;
  kgl::VertexArray _vao;
  kgl::Buffer _vbo;
  //! @}
};

bool App::_glfw_initialized = false;


int main()
{
  try
  {
    static constexpr auto width = 800;
    static constexpr auto height = 600;
    static constexpr auto title = "Hello Point Cloud";
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
