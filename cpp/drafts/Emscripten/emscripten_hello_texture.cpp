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

#include <DO/Sara/ImageProcessing/Flip.hpp>

#include <iostream>
#include <map>
#include <memory>

#ifdef EMSCRIPTEN
#  include <emscripten/emscripten.h>
#  define GLFW_INCLUDE_ES3
#endif

#include <GLFW/glfw3.h>


namespace sara = DO::Sara;

using namespace std;


Eigen::Matrix4d frustum(double l, double r, double b, double t, double n,
                        double f)
{
  auto proj = Eigen::Matrix4d{};

  // clang-format off
  proj <<
    2*n/(r-l),         0,  (r+l)/(r-l),            0,
            0, 2*n/(t-b),  (t+b)/(t-b),            0,
            0,         0, -(f+n)/(f-n), -2*f*n/(f-n),
            0,         0,           -1,            0;
  // clang-format on

  return proj;
}

Eigen::Matrix4d perspective(double fov, double aspect, double z_near,
                            double z_far)
{
  const auto t = z_near * std::tan(fov * M_PI / 360.);
  const auto b = -t;
  const auto l = aspect * b;
  const auto r = aspect * t;
  return frustum(l, r, b, t, z_near, z_far);
}

auto look_at(const Eigen::Vector3f& eye, const Eigen::Vector3f& center,
             const Eigen::Vector3f& up) -> Eigen::Matrix4f
{
  const Eigen::Vector3f f = (center - eye).normalized();
  Eigen::Vector3f u = up.normalized();
  const Eigen::Vector3f s = f.cross(u).normalized();
  u = s.cross(f);

  Eigen::Matrix4f res;
  // clang-format off
    res <<
       s.x(),  s.y(),  s.z(), -s.dot(eye),
       u.x(),  u.y(),  u.z(), -u.dot(eye),
      -f.x(), -f.y(), -f.z(),  f.dot(eye),
           0,      0,      0,           1;
  // clang-format on

  return res;
}


struct MyGLFW
{
  static GLFWwindow* window;
  static int width;
  static int height;

  static auto initialize() -> bool
  {
    if (glfwInit() != GL_TRUE)
    {
      std::cout << "Failed to initialize GLFW!" << std::endl;
      glfwTerminate();
      return false;
    }

    window = glfwCreateWindow(512, 512, "OpenGL Window", NULL, NULL);
    if (!MyGLFW::window)
    {
      std::cout << "Failed to create window!" << std::endl;
      glfwTerminate();
      return false;
    }

    glfwMakeContextCurrent(window);

    // Set the appropriate mouse and keyboard callbacks.
    glfwGetFramebufferSize(window, &width, &height);
    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetMouseButtonCallback(window, mouse_callback);
    glfwSetKeyCallback(window, key_callback);

    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;

    return true;
  }

  static void window_size_callback(GLFWwindow* /* window */, int width,
                                   int height)
  {
    std::cout << "window_size_callback received width: " << width
              << "  height: " << height << std::endl;
  }

  static void key_callback(GLFWwindow* /* window */, int key,
                           int /* scancode */, int action, int /* modifier */)
  {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
      glfwSetWindowShouldClose(window, 1);

    if (key == GLFW_KEY_ENTER)
      std::cout << "Hit Enter!" << std::endl;
  }

  static void mouse_callback(GLFWwindow* /* window */, int button,
                             int /* action */, int /* modifiers */)
  {
    std::cout << "Clicked mouse button: " << button << "!" << std::endl;
  }
};

GLFWwindow* MyGLFW::window = nullptr;
int MyGLFW::width = -1;
int MyGLFW::height = -1;


struct Scene
{
  // Host geometry data
  sara::Tensor_<float, 2> vertices;
  sara::Tensor_<std::uint32_t, 2> triangles;

  // OpenGL/Device geometry data.
  sara::GL::VertexArray vao;
  sara::GL::Buffer vbo;
  sara::GL::Buffer ebo;

  // OpenGL shaders.
  sara::GL::Shader vertex_shader;
  sara::GL::Shader fragment_shader;
  sara::GL::ShaderProgram shader_program;
  sara::GL::Texture2D texture;

  // Initialize the projection matrix once for all.
  Eigen::Matrix4f projection;
  Eigen::Matrix4f view;
  Eigen::Transform<float, 3, Eigen::Projective> transform;


  static std::unique_ptr<Scene> _scene;

  static auto instance() -> Scene&
  {
    if (_scene.get() == nullptr)
      throw std::runtime_error{"Please initialize the scene first"};
    return *_scene;
  }

  static auto initialize() -> void
  {
    if (_scene == nullptr)
      _scene.reset(new Scene);

    // Create a vertex shader.
    const std::map<std::string, int> arg_pos = {{"in_coords", 0},      //
                                                {"in_color", 1},       //
                                                {"in_tex_coords", 2},  //
                                                {"out_color", 0}};

    const auto vertex_shader_source = R"shader(#version 300 es
    layout (location = 0) in vec3 in_coords;
    layout (location = 1) in vec3 in_color;
    layout (location = 2) in vec2 in_tex_coords;

    uniform mat4 transform;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 out_color;
    out vec2 out_tex_coords;

    void main()
    {
      // gl_Position = projection * view * transform * vec4(in_coords, 1.0);
      gl_Position = vec4(in_coords, 1.0);
      out_color = in_color;
      out_tex_coords = in_tex_coords;
    }
    )shader";
    _scene->vertex_shader.create_from_source(GL_VERTEX_SHADER,
                                             vertex_shader_source);

    // Create a fragment shader.
    const auto fragment_shader_source = R"shader(#version 300 es
    #ifdef GL_ES
    precision highp float;
    #endif

    in vec3 out_color;
    in vec2 out_tex_coords;
    out vec4 frag_color;

    uniform sampler2D image;

    float sdf_line_segment(in vec2 p, in vec2 a, in vec2 b) {
      vec2 ba = b - a;
      vec2 pa = p - a;
      float t = clamp(dot(pa, ba) / dot(ba, ba), 0., 1.);
      return length(pa - t * ba);
    }

    void main()
    {
      vec2 a = vec2(0.2, 0.2);
      vec2 b = vec2(0.8, 0.8);
      vec2 c = vec2(0.3, 0.8);
      float thickness = 0.02;

      float d1 = sdf_line_segment(out_tex_coords, a, b) - thickness;
      float d2 = sdf_line_segment(out_tex_coords, b, c) - thickness;
      float d = min(d1, d2);

      vec4 out_color = texture(image, out_tex_coords) * vec4(out_color, 1.0);

      vec4 line_color = mix(vec4(1.0), out_color, 1. - smoothstep(.0, .015, abs(d)));

      if (d < 1e-7)
        frag_color = line_color;
      else
        frag_color = out_color;
    }
    )shader";
    _scene->fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                               fragment_shader_source);

    _scene->shader_program.create();
    _scene->shader_program.attach(_scene->vertex_shader,
                                  _scene->fragment_shader);

    _scene->vao.generate();
    _scene->vbo.generate();

    // Read the image from the disk.
    auto image = sara::imread<sara::Rgb8>("assets/sunflowerField.jpg");
    // Flip vertically so that the image data matches OpenGL image coordinate
    // system.
    sara::flip_vertically(image);

    // Copy the image to the GPU texture.
    _scene->texture.generate();
    _scene->texture.bind();
    _scene->texture.set_border_type(GL_CLAMP_TO_EDGE);
    _scene->texture.set_interpolation_type(GL_LINEAR);
    _scene->texture.initialize_data(image, 0);

    // Encode the vertex data in a tensor.
    _scene->vertices = sara::Tensor_<float, 2>{{4, 8}};
    // clang-format off
    _scene->vertices.flat_array() <<
      // coords            color              texture coords
       0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  1.0f, 0.0f,  // bottom-right
       0.5f,  0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  1.0f, 1.0f,  // top-right
      -0.5f,  0.5f, 0.0f,  1.0f, 1.0f, 0.0f,  0.0f, 1.0f,  // top-left
      -0.5f, -0.5f, 0.0f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f;  // bottom-left
    // clang-format on

    // Resize the quad vertices with the appropriate image ratio.
    const auto image_ratio = static_cast<float>(image.width()) / image.height();
    _scene->vertices.matrix().leftCols(1) *= image_ratio;

    _scene->triangles.resize(2, 3);
    // clang-format off
    _scene->triangles.flat_array() <<
      0, 1, 2,
      2, 3, 0;
    // clang-format on

    const auto row_bytes = [](const sara::TensorView_<float, 2>& data) {
      return data.size(1) * sizeof(float);
    };
    const auto float_pointer = [](int offset) {
      return reinterpret_cast<void*>(offset * sizeof(float));
    };

    _scene->vao.generate();

    // Vertex attributes.
    _scene->vbo.generate();

    // Triangles data.
    _scene->ebo.generate();

    {
      glBindVertexArray(_scene->vao);

      // Copy vertex data.
      _scene->vbo.bind_vertex_data(_scene->vertices);

      // Copy geometry data.
      _scene->ebo.bind_triangles_data(_scene->triangles);

      // Map the parameters to the argument position for the vertex shader.
      //
      // Vertex coordinates.
      glVertexAttribPointer(arg_pos.at("in_coords"), 3 /* 3D points */,
                            GL_FLOAT, GL_FALSE, row_bytes(_scene->vertices),
                            float_pointer(0));
      glEnableVertexAttribArray(arg_pos.at("in_coords"));

      // Colors.
      glVertexAttribPointer(arg_pos.at("in_color"), 3 /* 3D colors */, GL_FLOAT,
                            GL_FALSE, row_bytes(_scene->vertices),
                            float_pointer(3));
      glEnableVertexAttribArray(arg_pos.at("in_color"));

      // Texture coordinates.
      glVertexAttribPointer(arg_pos.at("in_tex_coords"), 2 /* 3D colors */,
                            GL_FLOAT, GL_FALSE, row_bytes(_scene->vertices),
                            float_pointer(6));
      glEnableVertexAttribArray(arg_pos.at("in_tex_coords"));
    }

    _scene->projection = perspective(45., 800. / 600., .1, 1000.).cast<float>();

    const Eigen::Vector3f position = 10.f * Eigen::Vector3f::UnitY();
    const Eigen::Vector3f front = -Eigen::Vector3f::UnitZ();
    const Eigen::Vector3f up = Eigen::Vector3f::UnitY();
    const Eigen::Vector3f world_up{Eigen::Vector3f::UnitY()};
    _scene->view = look_at(position, position + front, up);

    _scene->transform.setIdentity();
  }

  static auto destroy_opengl_data() -> void
  {
    _scene->vertex_shader.destroy();
    _scene->fragment_shader.destroy();
    _scene->vao.destroy();
    _scene->vbo.destroy();
    _scene->ebo.destroy();
    _scene->texture.destroy();
  }

  static auto render_frame()
  {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw triangles.
    const auto& scene = instance();
    glBindVertexArray(scene.vao);  // geometry specified by the VAO.
    glDrawElements(GL_TRIANGLES, scene.triangles.size(), GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(MyGLFW::window);
    glfwPollEvents();
  }
};

std::unique_ptr<Scene> Scene::_scene = nullptr;


struct PolylinePainter
{
  // A line can be thought a "thick" rectangular line. This thick line can then
  // be thought as a quad.
  //
  // In turn this quad can be decomposed into two right triangles adjacent at
  // their hypotenuse.
  std::vector<float> _vertices;
  std::vector<std::uint32_t> _triangles;
  float thickness;
  float antialias_radius;

  // OpenGL geometry data.
  sara::GL::VertexArray vao;
  sara::GL::Buffer vbo;
  sara::GL::Buffer ebo;

  // OpenGL Shader program.
  sara::GL::Shader vertex_shader;
  sara::GL::Shader fragment_shader;
  sara::GL::ShaderProgram shader_program;

  // Now the main task is generate the vertex coordinates from two points A and
  // B.
  auto add_line_segment(const Eigen::Vector2f& a, const Eigen::Vector2f& b)
      -> void
  {
    // First calculate the quad vertices.
    const Eigen::Vector2f t = (b - a).normalized();
    const Eigen::Vector2f o = Eigen::Vector2f(-t.y(), t.x());

    const auto& w = thickness;
    const auto& r = antialias_radius;
    const Eigen::Vector2f a0 = a - (w * 0.5f + r) * (t - o);
    const Eigen::Vector2f a1 = a - (w * 0.5f + r) * (t + o);
    const Eigen::Vector2f b0 = b + (w * 0.5f + r) * (t + o);
    const Eigen::Vector2f b1 = b + (w * 0.5f + r) * (t - o);
    _vertices.push_back(a0.x());
    _vertices.push_back(a0.y());
    _vertices.push_back(0.f);

    _vertices.push_back(a1.x());
    _vertices.push_back(a1.y());
    _vertices.push_back(0.f);

    _vertices.push_back(b0.x());
    _vertices.push_back(b0.y());
    _vertices.push_back(0.f);

    _vertices.push_back(b1.x());
    _vertices.push_back(b1.y());
    _vertices.push_back(0.f);

    const auto n = _triangles.size();
    _triangles.push_back(n + 0);  // a0
    _triangles.push_back(n + 2);  // b0
    _triangles.push_back(n + 1);  // a1

    _triangles.push_back(n + 2);  // b0
    _triangles.push_back(n + 3);  // b1
    _triangles.push_back(n + 1);  // a1
  }

  auto initialize() -> void
  {
    // Create a vertex shader.
    const std::map<std::string, int> arg_pos = {{"in_coords", 0},      //
                                                {"in_color", 1},       //
                                                {"in_tex_coords", 2},  //
                                                {"out_color", 0}};

    const auto vertex_shader_source = R"shader(#version 300 es
    layout (location = 0) in vec3 in_coords;
    layout (location = 1) in vec3 in_color;
    layout (location = 2) in vec2 in_tex_coords;

    out vec3 out_color;

    void main()
    {
      gl_Position = vec4(in_coords, 1.0);
      out_color = in_color;
    }
    )shader";
    vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);

    // Create a fragment shader.
    const auto fragment_shader_source = R"shader(#version 300 es
    #ifdef GL_ES
    precision highp float;
    #endif

    in vec3 out_color;
    out vec4 frag_color;

    uniform sampler2D image;

    float sdf_line_segment(in vec2 p, in vec2 a, in vec2 b) {
      vec2 ba = b - a;
      vec2 pa = p - a;
      float t = clamp(dot(pa, ba) / dot(ba, ba), 0., 1.);
      return length(pa - t * ba);
    }

    void main()
    {
      vec2 a = vec2(0.2, 0.2);
      vec2 b = vec2(0.8, 0.8);
      vec2 c = vec2(0.3, 0.8);
      float thickness = 0.02;

      float d1 = sdf_line_segment(out_tex_coords, a, b) - thickness;
      float d2 = sdf_line_segment(out_tex_coords, b, c) - thickness;
      float d = min(d1, d2);

      vec4 out_color = texture(image, out_tex_coords) * vec4(out_color, 1.0);

      vec4 line_color = mix(vec4(1.0), out_color, 1. - smoothstep(.0, .015, abs(d)));

      if (d < 1e-7)
        frag_color = line_color;
      else
        frag_color = out_color;
    }
    )shader";
    fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                       fragment_shader_source);

    shader_program.create();
    shader_program.attach(vertex_shader, fragment_shader);

#ifdef WIP
    vao.generate();
    vbo.generate();

    const auto row_bytes = [](const sara::TensorView_<float, 2>& data) {
      return data.size(1) * sizeof(float);
    };
    const auto float_pointer = [](int offset) {
      return reinterpret_cast<void*>(offset * sizeof(float));
    };

    vao.generate();

    // Vertex attributes.
    vbo.generate();

    // Triangles data.
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
      glVertexAttribPointer(arg_pos.at("in_coords"), 3 /* 3D points */,
                            GL_FLOAT, GL_FALSE, row_bytes(vertices),
                            float_pointer(0));
      glEnableVertexAttribArray(arg_pos.at("in_coords"));

      // Colors.
      glVertexAttribPointer(arg_pos.at("in_color"), 3 /* 3D colors */, GL_FLOAT,
                            GL_FALSE, row_bytes(vertices), float_pointer(3));
      glEnableVertexAttribArray(arg_pos.at("in_color"));

      // Texture coordinates.
      glVertexAttribPointer(arg_pos.at("in_tex_coords"), 2 /* 3D colors */,
                            GL_FLOAT, GL_FALSE, row_bytes(vertices),
                            float_pointer(6));
      glEnableVertexAttribArray(arg_pos.at("in_tex_coords"));
    }

    projection = perspective(45., 800. / 600., .1, 1000.).cast<float>();

    const Eigen::Vector3f position = 10.f * Eigen::Vector3f::UnitY();
    const Eigen::Vector3f front = -Eigen::Vector3f::UnitZ();
    const Eigen::Vector3f up = Eigen::Vector3f::UnitY();
    const Eigen::Vector3f world_up{Eigen::Vector3f::UnitY()};
    view = look_at(position, position + front, up);

    transform.setIdentity();
#endif
  }
};


int main()
{
  try
  {
    if (!MyGLFW::initialize())
      return EXIT_FAILURE;

    Scene::initialize();
    // Activate the texture 0 once for all.
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, Scene::instance().texture);

    // Specific rendering options.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);

    // We must use the shader program first before specifying the textures!
    Scene::instance().shader_program.use(true);

    // From then on, can we set the projective transformation.
    Scene::instance().shader_program.set_uniform_matrix4f(
        "transform", Scene::instance().transform.matrix().data());
    Scene::instance().shader_program.set_uniform_matrix4f(
        "view", Scene::instance().view.data());
    Scene::instance().shader_program.set_uniform_matrix4f(
        "projection", Scene::instance().projection.data());
    std::cout << Scene::instance().transform.matrix() << std::endl;
    std::cout << Scene::instance().view << std::endl;
    std::cout << Scene::instance().projection << std::endl;

    // Also specify the texture.
    const auto tex_location =
        glGetUniformLocation(Scene::instance().shader_program, "image");
    if (tex_location == GL_INVALID_VALUE)
      throw std::runtime_error{"Cannot find texture location!"};
    glUniform1i(tex_location, 0);

#  ifdef EMSCRIPTEN
    emscripten_set_main_loop(Scene::render_frame, 0, 1);
#  else
    while (!glfwWindowShouldClose(MyGLFW::window))
      Scene::render_frame();
#  endif

    Scene::destroy_opengl_data();

    glfwTerminate();
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
