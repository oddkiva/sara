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

#include <fmt/format.h>

#ifdef _WIN32
#  include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <cmath>
#include <map>

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"


namespace kgl = DO::Kalpana::GL;

using namespace DO::Sara;
using namespace std;


struct ShaderProgramBuilder
{
  const std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                              {"in_color", 1},   //
                                              {"out_color", 0}};

  auto row_bytes(const TensorView_<float, 2>& data) const -> GLsizei
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
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;

    float dist = length(gl_PointCoord - vec2(0.5));

    if (dot(circCoord, circCoord) > 1.0)
        discard;
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

    // Prepare OpenGL first before any OpenGL calls.
    init_opengl();

    init_shader_program();
    init_data_on_opengl();
    init_render_settings();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    // ImGuiIO &io = ImGui::GetIO();
    ImGui_ImplGlfw_InitForOpenGL(_window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
    ImGui::StyleColorsDark();
  }

  ~App()
  {
    _vbo.destroy();
    _vao.destroy();
    _shader_program.clear();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

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
      glfwPollEvents();

      // Start the Dear ImGui frame
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      {
        int bar_data[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        static constexpr auto n = 1000;
        float x_data[n];
        float y_data[n];
        for (auto i = 0; i < n; ++i)
        {
          x_data[i] = static_cast<float>(i) / n * 2 * M_PI;
          y_data[i] = std::sin(x_data[i]);
        }

        ImGui::Begin("Graphs with ImPlot");
        if (ImPlot::BeginPlot("My Plot"))
        {
          ImPlot::PlotBars("My Bar Plot", bar_data, 11);
          ImPlot::PlotLine("My Line Plot", x_data, y_data, n);
          // ...
          ImPlot::EndPlot();
        }
        ImGui::End();
      }
      ImGui::Render();


      // Draw triangles
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glBindVertexArray(_vao);
      glDrawArrays(GL_POINTS, 0, _num_vertices);

      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      glfwSwapBuffers(_window);
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
    auto vertices = Tensor_<float, 2>{{3, 6}};
    // clang-format off
    vertices.flat_array() <<
      // coords            color
      -0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  // left
       0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  // right
       0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f;  // top
    // clang-format on
    _num_vertices = vertices.size(0);

    _vao = kgl::VertexArray{};
    _vao.generate();

    _vbo = kgl::Buffer{};
    _vbo.generate();

    const auto shader_program_builder = ShaderProgramBuilder{};
    const auto& shader_arg_pos = shader_program_builder.arg_pos;

    // Specify the vertex attributes here.
    glBindVertexArray(_vao);

    // Copy the vertex data into the GPU buffer object.
    _vbo.bind_vertex_data(vertices);

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
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
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
      throw std::runtime_error{
          fmt::format("Error: failed to initialize GLEW: {}",
                      reinterpret_cast<const char*>(glewGetErrorString(err)))};
#endif
  }

private:
  static bool _glfw_initialized;

  GLFWwindow* _window = nullptr;

  // The graphics pipeline.
  kgl::ShaderProgram _shader_program;

  // Geometry data on OpenGL side.
  GLsizei _num_vertices;
  kgl::VertexArray _vao;
  kgl::Buffer _vbo;
};

bool App::_glfw_initialized = false;


auto main() -> int
{
  try
  {
    static constexpr auto width = 800;
    static constexpr auto height = 600;
    static constexpr auto title = "Hello Triangle";

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
