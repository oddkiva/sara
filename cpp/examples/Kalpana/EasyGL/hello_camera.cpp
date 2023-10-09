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
#include <DO/Kalpana/EasyGL/Objects/Camera.hpp>
#include <DO/Kalpana/EasyGL/TrackBall.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>

#include <GLFW/glfw3.h>


#include "Time.hpp"

#include <map>


namespace k = DO::Kalpana;
namespace kgl = k::GL;
namespace sara = DO::Sara;


auto read_point_cloud(const std::string& h5_filepath) -> sara::Tensor_<float, 2>
{
  auto h5_file = sara::H5File{h5_filepath, H5F_ACC_RDONLY};

  auto coords = Eigen::MatrixXd{};
  h5_file.read_dataset("points", coords);
  coords.matrix() *= -1;
  auto coords_tensorview = sara::TensorView_<double, 2>{
      coords.data(), {coords.cols(), coords.rows()}};

  auto colors = sara::Tensor_<double, 2>{};
  h5_file.read_dataset("colors", colors);

  // Concatenate the data.
  auto vertex_data = sara::Tensor_<double, 2>{{coords.cols(), 6}};
  vertex_data.matrix() << coords_tensorview.matrix(), colors.matrix();

  return vertex_data.cast<float>();
}

auto make_point_cloud()
{
  // Encode the vertex data in a tensor.
#ifdef __APPLE__
  const auto vertex_data = read_point_cloud("/Users/david/Desktop/geometry.h5");
#else
  const auto vertex_data = read_point_cloud("/home/david/Desktop/geometry.h5");
#endif
  SARA_DEBUG << "vertices =\n" << vertex_data.matrix().topRows(20) << std::endl;
  return vertex_data;
}


struct PointCloudObject
{
  PointCloudObject(const sara::Tensor_<float, 2>& vertices_)
    : vertices{vertices_}
  {
    shader_program = make_shader();

    // =========================================================================
    // Encode the vertex data in a tensor.
    //
    const auto row_bytes = [](const sara::TensorView_<float, 2>& data) {
      return static_cast<GLsizei>(data.size(1) * sizeof(float));
    };
    const auto float_pointer = [](int offset) {
      return reinterpret_cast<void*>(offset * sizeof(float));
    };

    vao.generate();


    // =========================================================================
    // Setup Vertex attributes on the GPU side.
    //
    vbo.generate();
    {
      glBindVertexArray(vao);

      // Copy vertex data.
      vbo.bind_vertex_data(vertices);

      // Map the parameters to the argument position for the vertex shader.
      //
      // Vertex coordinates.
      glVertexAttribPointer(arg_pos["in_coords"], 3 /* 3D points */, GL_FLOAT,
                            GL_FALSE, row_bytes(vertices), float_pointer(0));
      glEnableVertexAttribArray(arg_pos["in_coords"]);

      // Texture coordinates.
      glVertexAttribPointer(arg_pos["in_color"], 3 /* RGB_COLOR */, GL_FLOAT,
                            GL_FALSE, row_bytes(vertices), float_pointer(3));
      glEnableVertexAttribArray(arg_pos["in_color"]);
    }
  }

  auto make_shader() -> kgl::ShaderProgram
  {
    const auto vertex_shader_source = R"shader(
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


    const auto fragment_shader_source = R"shader(
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

  auto destroy() -> void
  {
    vao.destroy();
    vbo.destroy();
  }

  sara::Tensor_<float, 2> vertices;
  kgl::Buffer vbo;
  kgl::VertexArray vao;
  kgl::ShaderProgram shader_program;
  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                        {"in_color", 1}};
};


struct CheckerBoardObject
{
  CheckerBoardObject(int rows_ = 20, int cols_ = 20, float scale = 10.f)
    : rows{rows_}
    , cols{cols_}
  {
    shader_program = make_shader();

    vertices = sara::Tensor_<float, 2>{{4 * rows * cols, 6}};
    triangles = sara::Tensor_<unsigned int, 2>{{2 * rows * cols, 3}};

    auto v_mat = vertices.matrix();
    auto t_mat = triangles.matrix();
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
      {
        const auto ij = cols * i + j;

        // Coordinates.
        //
        // clang-format off
        v_mat.block(4 * ij, 0, 4, 3) <<  // coords
            i + 0.5f, 0.0f, j + 0.5f,    // top-right
            i + 0.5f, 0.0f, j + -0.5f,   // bottom-right
            i + -0.5f, 0.0f, j + -0.5f,  // bottom-left
            i + -0.5f, 0.0f, j + 0.5f;   // top-left
        // clang-format on

        // Set colors.
        if (i % 2 == 0 && j % 2 == 0)
          v_mat.block(4 * ij, 3, 4, 3).setZero();
        else if (i % 2 == 0 && j % 2 == 1)
          v_mat.block(4 * ij, 3, 4, 3).setOnes();
        else if (i % 2 == 1 && j % 2 == 0)
          v_mat.block(4 * ij, 3, 4, 3).setOnes();
        else  // (i % 2 == 1 and j % 2 == 0)
          v_mat.block(4 * ij, 3, 4, 3).setZero();

        // vertex indices for each triangle that forms the quad
        //
        // clang-format off
        t_mat.block(2 * ij, 0, 2, 3) <<
          4 * ij + 0, 4 * ij + 1, 4 * ij + 2,
          4 * ij + 2, 4 * ij + 3, 4 * ij + 0;
        // clang-format on
      }
    }
    // Translate.
    v_mat.col(0).array() -= rows / 2.f;
    v_mat.col(2).array() -= cols / 2.f;
    // Rescale.
    v_mat.leftCols(3) *= scale;

    const auto row_bytes = [](const sara::TensorView_<float, 2>& data) {
      return static_cast<GLsizei>(data.size(1) * sizeof(float));
    };
    const auto float_pointer = [](int offset) {
      return reinterpret_cast<void*>(offset * sizeof(float));
    };

    vao.generate();
    vbo.generate();
    ebo.generate();

    // Specify the vertex attributes here.
    {
      glBindVertexArray(vao);

      // Copy the vertex data into the GPU buffer object.
      vbo.bind_vertex_data(vertices);

      // Copy the triangles data into the GPU buffer object.
      ebo.bind_triangles_data(triangles);

      // Specify that the vertex shader param 0 corresponds to the first 3 float
      // data of the buffer object.
      glVertexAttribPointer(arg_pos.at("in_coords"), 3 /* 3D points */,
                            GL_FLOAT, GL_FALSE, row_bytes(vertices),
                            float_pointer(0));
      glEnableVertexAttribArray(arg_pos.at("in_coords"));

      // Specify that the vertex shader param 1 corresponds to the first 3 float
      // data of the buffer object.
      glVertexAttribPointer(arg_pos.at("in_color"), 3 /* 3D colors */, GL_FLOAT,
                            GL_FALSE, row_bytes(vertices), float_pointer(3));
      glEnableVertexAttribArray(arg_pos.at("in_color"));

      // Unbind the vbo to protect its data.
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glBindVertexArray(0);
    }
  }

  auto make_shader() -> kgl::ShaderProgram
  {
    const auto vertex_shader_source = R"shader(
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
    gl_PointSize = 200.0;
    out_color = in_color;
  }
  )shader";
    auto vertex_shader = kgl::Shader{};
    vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);


    const auto fragment_shader_source = R"shader(
#version 330 core
  in vec3 out_color;
  out vec4 frag_color;

  void main()
  {
    frag_color = vec4(out_color, 0.1);
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

  auto destroy() -> void
  {
    vao.destroy();
    vbo.destroy();
    ebo.destroy();
  }

  int rows{100};
  int cols{100};
  sara::Tensor_<float, 2> vertices;
  sara::Tensor_<unsigned int, 2> triangles;
  kgl::Buffer vbo;
  kgl::Buffer ebo;
  kgl::VertexArray vao;
  kgl::ShaderProgram shader_program;
  const std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                              {"in_color", 1},   //
                                              {"out_color", 0}};
};


struct Scene;
class App;


class App
{
public:
  App(const Eigen::Vector2i& sizes, const std::string& title)
  {
    init_glfw();

    _window = glfwCreateWindow(sizes.x(), sizes.y(),  //
                               title.c_str(),         //
                               nullptr, nullptr);
    glfwMakeContextCurrent(_window);
    glfwSetWindowUserPointer(_window, this);
    glfwSetFramebufferSizeCallback(_window, resize_framebuffer);
    glfwSetKeyCallback(_window, move_camera_from_keyboard);
    glfwSetCursorPosCallback(_window, move_trackball);
    glfwSetMouseButtonCallback(_window, use_trackball);

    init_opengl();
  }

  ~App()
  {
    if (_window)
      glfwDestroyWindow(_window);

    if (_glfw_initialized)
      glfwTerminate();
  }

  auto run() -> void;

private:
  auto init_opengl() -> void
  {
    // GLFW context...
    glfwMakeContextCurrent(_window);

    // Init OpenGL extensions.
    init_glew();
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
  static auto get_self(GLFWwindow* const window) -> App&
  {
    const auto app_void_ptr = glfwGetWindowUserPointer(window);
    if (app_void_ptr == nullptr)
      throw std::runtime_error{
          "Please call glfwSetWindowUserPointer to register this window!"};
    const auto app_ptr = reinterpret_cast<App*>(app_void_ptr);
    return *app_ptr;
  }

  static auto resize_framebuffer(GLFWwindow*, int width, int height) -> void
  {
    // make sure the viewport matches the new window dimensions; note that width
    // and height will be significantly larger than specified on retina
    // displays.
    glViewport(0, 0, width, height);
  }

  static auto move_camera_from_keyboard(GLFWwindow* window,  //
                                        int key,             //
                                        int /* scancode */,  //
                                        int action,          //
                                        int /* mods */) -> void
  {
    if (action == GLFW_RELEASE)
      return;

    auto& app = get_self(window);

    if (key == GLFW_KEY_W)
      app.camera.move_forward(app.gtime.delta_time);
    if (key == GLFW_KEY_S)
      app.camera.move_backward(app.gtime.delta_time);
    if (key == GLFW_KEY_A)
      app.camera.move_left(app.gtime.delta_time);
    if (key == GLFW_KEY_D)
      app.camera.move_right(app.gtime.delta_time);

    if (key == GLFW_KEY_DELETE)
      app.camera.no_head_movement(-app.gtime.delta_time);  // CCW
    if (glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
      app.camera.no_head_movement(+app.gtime.delta_time);  // CW

    if (key == GLFW_KEY_HOME)
      app.camera.yes_head_movement(+app.gtime.delta_time);
    if (key == GLFW_KEY_END)
      app.camera.yes_head_movement(-app.gtime.delta_time);

    if (key == GLFW_KEY_R)
      app.camera.move_up(app.gtime.delta_time);
    if (key == GLFW_KEY_F)
      app.camera.move_down(app.gtime.delta_time);

    if (key == GLFW_KEY_INSERT)
      app.camera.maybe_head_movement(-app.gtime.delta_time);
    if (key == GLFW_KEY_PAGE_UP)
      app.camera.maybe_head_movement(+app.gtime.delta_time);

    if (key == GLFW_KEY_SPACE)
      app.show_checkerboard = !app.show_checkerboard;

    if (key == GLFW_KEY_MINUS)
      app.scale /= scale_factor;
    if (key == GLFW_KEY_EQUAL)
      app.scale *= scale_factor;

    app.camera.update();
  }

  static auto use_trackball(GLFWwindow* window, int button, int action,
                            int /* mods */) -> void
  {
    if (button != GLFW_MOUSE_BUTTON_LEFT)
      return;

    auto& app = get_self(window);

    auto p = Eigen::Vector2d{};
    glfwGetCursorPos(window, &p.x(), &p.y());

    const Eigen::Vector2f pf = normalize_cursor_pos(window, p).cast<float>();
    if (action == GLFW_PRESS && !app.trackball.pressed())
      app.trackball.push(pf);
    else if (action == GLFW_RELEASE && app.trackball.pressed())
      app.trackball.release(pf);
  }

  static auto move_trackball(GLFWwindow* window, double x, double y) -> void
  {
    const auto curr_pos = Eigen::Vector2d{x, y};
    const Eigen::Vector2f p =
        normalize_cursor_pos(window, curr_pos).cast<float>();

    auto& app = get_self(window);

    if (app.trackball.pressed())
      app.trackball.move(p);
  }

  static inline auto normalize_cursor_pos(GLFWwindow* window,
                                          const Eigen::Vector2d& pos)
      -> Eigen::Vector2d
  {
    auto w = int{};
    auto h = int{};
    glfwGetWindowSize(window, &w, &h);

    const Eigen::Vector2d c = Eigen::Vector2i(w, h).cast<double>() * 0.5;

    Eigen::Vector2d normalized_pos = ((pos - c).array() / c.array()).matrix();
    normalized_pos.y() *= -1;
    return normalized_pos;
  };

public:
  static bool _glfw_initialized;
  GLFWwindow* _window = nullptr;

  k::Camera camera = {};
  Time gtime = Time{};

  DO::Kalpana::GL::TrackBall trackball = {};

  bool show_checkerboard = true;
  float scale = 1.f;
  static constexpr auto scale_factor = 1.05f;
};

auto App::_glfw_initialized = false;


struct Scene
{
  auto init() -> void
  {
    // Setup options for point cloud rendering.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // You absolutely need this for 3D objects!
    glEnable(GL_DEPTH_TEST);

    // Background color.
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    glfwSwapInterval(1);
  }

  auto run(App& app) -> void
  {
    // Initialize the projection matrix once for all.
    const auto projection = k::perspective(45.f, 800.f / 600.f, .1f, 1000.f);

    // Transform matrix.
    const Eigen::Transform<float, 3, Eigen::Projective> transform =
        Eigen::Transform<float, 3, Eigen::Projective>::Identity();

    const auto view_uniform =
        checkerboard.shader_program.get_uniform_location("view");
    const auto proj_uniform =
        checkerboard.shader_program.get_uniform_location("projection");
    const auto tsfm_uniform =
        checkerboard.shader_program.get_uniform_location("transform");

    while (!glfwWindowShouldClose(app._window))
    {
      // Calculate the elapsed time.
      app.gtime.update();

      if (glfwGetKey(app._window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(app._window, true);

      // Camera interaction with _windowwackball.
      // auto view_matrix = camera.view_matrix();
      Eigen::Matrix3f view_matrix_33 =
          app.trackball.rotation().toRotationMatrix().cast<float>();
      Eigen::Matrix4f view_matrix = Eigen::Matrix4f::Identity();
      view_matrix.topLeftCorner(3, 3) = view_matrix_33;
      view_matrix.col(3).head(3) = app.camera.position;

      Eigen::Transform<float, 3, Eigen::Projective> scale_point_cloud =
          Eigen::Transform<float, 3, Eigen::Projective>::Identity();
      scale_point_cloud.scale(app.scale);

      // Important.
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      // Draw the checkerboard.
      if (app.show_checkerboard)
      {
        checkerboard.shader_program.use();
        checkerboard.shader_program.set_uniform_matrix4f(
            tsfm_uniform, transform.matrix().data());
        checkerboard.shader_program.set_uniform_matrix4f(view_uniform,
                                                         view_matrix.data());
        checkerboard.shader_program.set_uniform_matrix4f(proj_uniform,
                                                         projection.data());
        glBindVertexArray(checkerboard.vao);
        glDrawElements(GL_TRIANGLES,
                       static_cast<GLsizei>(checkerboard.triangles.size()),
                       GL_UNSIGNED_INT, 0);
      }

      // Draw point cloud.
      point_cloud_object.shader_program.use();
      point_cloud_object.shader_program.set_uniform_matrix4f(
          tsfm_uniform, scale_point_cloud.matrix().data());
      point_cloud_object.shader_program.set_uniform_matrix4f(
          view_uniform, view_matrix.data());
      point_cloud_object.shader_program.set_uniform_matrix4f(proj_uniform,
                                                             projection.data());

      glBindVertexArray(point_cloud_object.vao);
      glDrawArrays(GL_POINTS, 0, point_cloud_object.vertices.size(0));

      glfwSwapBuffers(app._window);
      glfwPollEvents();
    }
  }

  auto destroy() -> void
  {
    point_cloud_object.destroy();
    checkerboard.destroy();
  }

  PointCloudObject point_cloud_object = PointCloudObject{make_point_cloud()};
  CheckerBoardObject checkerboard = CheckerBoardObject{20, 20, 10.f};
};


auto App::run() -> void
{
  auto scene = Scene{};
  scene.init();
  scene.run(*this);
  scene.destroy();
}


auto main() -> int
{
  try
  {
    // Create a window.
    static constexpr auto width = 800;
    static constexpr auto height = 600;
    static constexpr auto title = "Hello Camera";

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
