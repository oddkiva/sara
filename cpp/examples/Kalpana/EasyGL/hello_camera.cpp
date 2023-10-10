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
#include <DO/Kalpana/EasyGL/Objects/ColoredPointCloud.hpp>
#include <DO/Kalpana/EasyGL/Renderer/CheckerboardRenderer.hpp>
#include <DO/Kalpana/EasyGL/Renderer/ColoredPointCloudRenderer.hpp>
#include <DO/Kalpana/EasyGL/TrackBall.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>

#include <GLFW/glfw3.h>

#include <map>


namespace k = DO::Kalpana;
namespace kgl = k::GL;
namespace sara = DO::Sara;


class Scene;
struct UserInteractionResponder;
class App;


struct UserInteractionResponder
{
  struct Time
  {
    void update()
    {
      last_frame = current_frame;
      current_frame = static_cast<float>(timer.elapsed_ms());
      delta_time = current_frame - last_frame;
    }

    DO::Sara::Timer timer;
    float delta_time = 0.f;
    float last_frame = 0.f;
    float current_frame = 0.f;
  };

  Time gtime = Time{};

  //! @brief View objects.
  //! @{
  k::Camera camera = {};
  kgl::TrackBall trackball = {};
  //! @}

  //! @brief View parameters.
  //! @{
  bool show_checkerboard = true;
  float scale = 1.f;
  static constexpr auto scale_factor = 1.05f;
  //! @}

  auto normalize_cursor_pos(GLFWwindow* const window,
                            const Eigen::Vector2d& pos) const -> Eigen::Vector2d
  {
    auto w = int{};
    auto h = int{};
    glfwGetWindowSize(window, &w, &h);

    const Eigen::Vector2d c = Eigen::Vector2i(w, h).cast<double>() * 0.5;

    Eigen::Vector2d normalized_pos = ((pos - c).array() / c.array()).matrix();
    normalized_pos.y() *= -1;
    return normalized_pos;
  };


  auto key_pressed(GLFWwindow* const window, const int key, const int action)
      -> void
  {
    if (action == GLFW_RELEASE)
      return;

    if (key == GLFW_KEY_W)
      camera.move_forward(gtime.delta_time);
    if (key == GLFW_KEY_S)
      camera.move_backward(gtime.delta_time);
    if (key == GLFW_KEY_A)
      camera.move_left(gtime.delta_time);
    if (key == GLFW_KEY_D)
      camera.move_right(gtime.delta_time);

    if (key == GLFW_KEY_DELETE)
      camera.no_head_movement(-gtime.delta_time);  // CCW
    if (glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS)
      camera.no_head_movement(+gtime.delta_time);  // CW

    if (key == GLFW_KEY_HOME)
      camera.yes_head_movement(+gtime.delta_time);
    if (key == GLFW_KEY_END)
      camera.yes_head_movement(-gtime.delta_time);

    if (key == GLFW_KEY_R)
      camera.move_up(gtime.delta_time);
    if (key == GLFW_KEY_F)
      camera.move_down(gtime.delta_time);

    if (key == GLFW_KEY_INSERT)
      camera.maybe_head_movement(-gtime.delta_time);
    if (key == GLFW_KEY_PAGE_UP)
      camera.maybe_head_movement(+gtime.delta_time);

    if (key == GLFW_KEY_SPACE)
      show_checkerboard = !show_checkerboard;

    if (key == GLFW_KEY_MINUS)
      scale /= scale_factor;
    if (key == GLFW_KEY_EQUAL)
      scale *= scale_factor;

    camera.update();
  }

  auto mouse_pressed(GLFWwindow* const window, const int button,
                     const int action) -> void
  {
    if (button != GLFW_MOUSE_BUTTON_LEFT)
      return;

    auto p = Eigen::Vector2d{};
    glfwGetCursorPos(window, &p.x(), &p.y());

    const Eigen::Vector2f pf = normalize_cursor_pos(window, p).cast<float>();
    if (action == GLFW_PRESS && !trackball.pressed())
      trackball.push(pf);
    else if (action == GLFW_RELEASE && trackball.pressed())
      trackball.release(pf);
  }

  auto mouse_moved(GLFWwindow* const window, const double x, const double y)
      -> void
  {
    const auto curr_pos = Eigen::Vector2d{x, y};
    const Eigen::Vector2f p =
        normalize_cursor_pos(window, curr_pos).cast<float>();

    if (trackball.pressed())
      trackball.move(p);
  }
};


class Scene
{
public:
  auto init() -> void
  {
    _point_cloud.initialize();
    _point_cloud.upload_host_data_to_gl(make_point_cloud());
    _point_cloud_renderer.initialize();

    _checkerboard = kgl::CheckerboardRenderer{20, 20, 10.f, 0.f};
    _checkerboard.initialize();

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

  auto run(GLFWwindow* const window, UserInteractionResponder& responder)
      -> void
  {
    // Initialize the projection matrix once for all.
    const auto projection = k::perspective(45.f, 800.f / 600.f, .1f, 1000.f);

    // Transform matrix.
    const Eigen::Transform<float, 3, Eigen::Projective> transform =
        Eigen::Transform<float, 3, Eigen::Projective>::Identity();

    const auto view_uniform = _checkerboard._view_loc;
    const auto proj_uniform = _checkerboard._projection_loc;
    const auto tsfm_uniform = _checkerboard._transform_loc;

    while (!glfwWindowShouldClose(window))
    {
      // Calculate the elapsed time.
      responder.gtime.update();

      if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

      // Camera interaction with _windowwackball.
      // auto view_matrix = camera.view_matrix();
      Eigen::Matrix3f view_matrix_33 =
          responder.trackball.rotation().toRotationMatrix().cast<float>();
      Eigen::Matrix4f view_matrix = Eigen::Matrix4f::Identity();
      view_matrix.topLeftCorner(3, 3) = view_matrix_33;
      view_matrix.col(3).head(3) = responder.camera.position;

      Eigen::Transform<float, 3, Eigen::Projective> scale_point_cloud =
          Eigen::Transform<float, 3, Eigen::Projective>::Identity();
      scale_point_cloud.scale(responder.scale);

      // Important.
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      // Draw the checkerboard.
      if (responder.show_checkerboard)
        _checkerboard.render(transform.matrix(), view_matrix, projection);

        // Draw point cloud.
      _point_cloud_renderer.render(_point_cloud, 2.f,
                                   scale_point_cloud.matrix(), view_matrix,
                                   projection);

      glfwSwapBuffers(window);
      glfwPollEvents();
    }
  }

  auto destroy() -> void
  {
    _point_cloud.destroy();
    _point_cloud_renderer.destroy();
    _checkerboard.destroy();
  }

private:
  auto read_point_cloud(const std::string& h5_filepath) const
      -> sara::Tensor_<float, 2>
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

  auto make_point_cloud() const -> sara::Tensor_<float, 2>
  {
    // Encode the vertex data in a tensor.
#ifdef __APPLE__
    const auto vertex_data =
        read_point_cloud("/Users/david/Desktop/geometry.h5");
#else
    const auto vertex_data =
        read_point_cloud("/home/david/Desktop/geometry.h5");
#endif
    SARA_DEBUG << "vertices =\n"
               << vertex_data.matrix().topRows(20) << std::endl;
    return vertex_data;
  }

  kgl::ColoredPointCloud _point_cloud;
  kgl::ColoredPointCloudRenderer _point_cloud_renderer;
  kgl::CheckerboardRenderer _checkerboard;
};


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

    _scene.init();
  }

  ~App()
  {
    _scene.destroy();

    if (_window)
      glfwDestroyWindow(_window);

    if (_glfw_initialized)
      glfwTerminate();
  }

  auto run() -> void
  {
    _scene.run(_window, _responder);
  }

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

  static auto get_self(GLFWwindow* const window) -> App&
  {
    const auto app_void_ptr = glfwGetWindowUserPointer(window);
    if (app_void_ptr == nullptr)
      throw std::runtime_error{
          "Please call glfwSetWindowUserPointer to register this window!"};
    const auto app_ptr = reinterpret_cast<App*>(app_void_ptr);
    return *app_ptr;
  }

private: /* callbacks */
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
    auto& app = get_self(window);
    app._responder.key_pressed(window, key, action);
  }

  static auto use_trackball(GLFWwindow* window, int button, int action,
                            int /* mods */) -> void
  {
    auto& app = get_self(window);
    app._responder.mouse_pressed(window, button, action);
  }

  static auto move_trackball(GLFWwindow* window, double x, double y) -> void
  {
    auto& app = get_self(window);
    app._responder.mouse_moved(window, x, y);
  }

public:
  static bool _glfw_initialized;
  GLFWwindow* _window = nullptr;

  UserInteractionResponder _responder;
  Scene _scene;
};

auto App::_glfw_initialized = false;


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
