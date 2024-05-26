// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Kalpana/EasyGL.hpp>
#include <DO/Kalpana/EasyGL/Objects/Camera.hpp>
#include <DO/Kalpana/EasyGL/Objects/ColoredPointCloud.hpp>
#include <DO/Kalpana/EasyGL/Objects/TexturedImage.hpp>
#include <DO/Kalpana/EasyGL/Objects/TexturedQuad.hpp>
#include <DO/Kalpana/EasyGL/Renderer/ColoredPointCloudRenderer.hpp>
#include <DO/Kalpana/EasyGL/Renderer/TextureRenderer.hpp>
#include <DO/Kalpana/Math/Projection.hpp>
#include <DO/Kalpana/Math/Viewport.hpp>

#include <DO/Sara/Logging/Logger.hpp>
#include <DO/Sara/SfM/Odometry/OdometryPipeline.hpp>

#if defined(_WIN32)
#  include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <fmt/format.h>

#include <string_view>

#if defined(_OPENMP)
#  include <omp.h>
#endif


namespace fs = std::filesystem;
namespace sara = DO::Sara;
namespace k = DO::Kalpana;
namespace kgl = DO::Kalpana::GL;

using sara::operator""_m;
using sara::operator""_deg;


class SingleWindowApp
{
public:
  SingleWindowApp(const Eigen::Vector2i& sizes, const std::string& title)
  {
    // Init GLFW.
    init_glfw();

    // Create a window.
    _window = glfwCreateWindow(sizes.x(), sizes.y(),  //
                               title.c_str(),         //
                               nullptr, nullptr);

    _fb_sizes = get_framebuffer_sizes();

    // Initialize the point cloud viewport.
    _point_cloud_viewport.top_left().setZero();
    _point_cloud_viewport.sizes() << _fb_sizes.x() / 2, _fb_sizes.y();

    // Initialize the video viewport.
    _video_viewport.top_left() << _fb_sizes.x() / 2, 0;
    _video_viewport.sizes() << _fb_sizes.x() / 2, _fb_sizes.y();

    // Prepare OpenGL first before any OpenGL calls.
    init_opengl();

    // The magic function.
    glfwSetWindowUserPointer(_window, this);
    // Register callbacks.
    glfwSetWindowSizeCallback(_window, window_size_callback);
    glfwSetKeyCallback(_window, key_callback);
  }

  ~SingleWindowApp()
  {
    // Destroy GL objects.
    deinit_gl_resources();

    // Destroy GLFW.
    if (_window != nullptr)
      glfwDestroyWindow(_window);

    if (_glfw_initialized)
      glfwTerminate();
  }

  auto set_config(const fs::path& video_path,
                  const sara::v2::BrownConradyDistortionModel<double>& camera)
      -> void
  {
    _pipeline.set_config(video_path, camera);
    init_gl_resources();
  }

  auto run() -> void
  {
    // Current projection matrix
    _projection = _video_viewport.orthographic_projection();
    // _point_cloud_projection = _point_cloud_viewport.orthographic_projection();
    _point_cloud_projection = _point_cloud_viewport.perspective(120.f, 1e-6f, 1e3f);

    // Background color.
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // You absolutely need this for 3D objects!
    glEnable(GL_DEPTH_TEST);

    // Display image.
    glfwSwapInterval(1);
    while (!glfwWindowShouldClose(_window))
    {
      if (!_pause)
      {
        if (_quit)
          break;

        if (!_pipeline.read())
          break;

        if (!_pipeline._video_streamer.skip())
        {
          _pipeline.process();

          // Load data to OpenGL.
          //
          // TODO: upload only if we have a new image frame to process and only
          // if the absolute pose estimation is successful.
          upload_point_cloud_data_to_opengl();

          _pause = true;
        }
      }

      if (_quit)
        break;

      // Clear the color buffer and the buffer testing.
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      render_video();
      render_point_cloud();

      glfwSwapBuffers(_window);
      glfwPollEvents();
    }
  }

private:
  auto init_opengl() -> void
  {
    glfwMakeContextCurrent(_window);
    init_glew();
  }

  auto init_gl_resources() -> void
  {
    // Video texture rendering
    _texture.initialize(_pipeline._video_streamer.frame_rgb8(), 0);

    const auto& w = _pipeline._video_streamer.width();
    const auto& h = _pipeline._video_streamer.height();
    const auto aspect_ratio = static_cast<float>(w) / h;
    auto vertices = _quad.host_vertices().matrix();
    vertices.col(0) *= aspect_ratio;
    _quad.initialize();

    _texture_renderer.initialize();

    // Point cloud rendering
    _point_cloud.initialize();
    _point_cloud_renderer.initialize();
  }

  auto deinit_gl_resources() -> void
  {
    _texture.destroy();
    _quad.destroy();
    _texture_renderer.destroy();

    _point_cloud.destroy();
    _point_cloud_renderer.destroy();
  }

  auto upload_point_cloud_data_to_opengl() -> void
  {
    const auto& point_cloud = _pipeline.point_cloud();

    static constexpr auto dim = 6;
    const auto num_points = static_cast<int>(point_cloud.size());
    if (num_points == 0)
      return;

    const auto ptr =
        const_cast<sara::PointCloudGenerator::ScenePoint*>(point_cloud.data());
    const auto ptrd = reinterpret_cast<double*>(ptr);
    const auto pc_tview = sara::TensorView_<double, 2>{
        ptrd,              //
        {num_points, dim}  //
    };

    auto& logger = sara::Logger::get();
    SARA_LOGI(logger, "point cloud dimensions: {} ",
              pc_tview.sizes().transpose().eval());
    _point_cloud.upload_host_data_to_gl(pc_tview.cast<float>());
  }

  auto render_video() -> void
  {
    // Render on the right half of the window surface.
    glViewport(_video_viewport.x(), _video_viewport.y(),  //
               _video_viewport.width(), _video_viewport.height());
    // Transfer the CPU image frame data to the OpenGL texture.
    _texture.reset(_pipeline.make_display_frame());
    // Render the texture on the quad.
    auto model_view = Eigen::Transform<float, 3, Eigen::Projective>{};
    model_view.setIdentity();
    _texture_renderer.render(_texture, _quad, model_view.matrix(), _projection);
  }

  auto render_point_cloud() -> void
  {
    if (_pipeline.point_cloud().empty())
      return;

    glViewport(_point_cloud_viewport.x(), _point_cloud_viewport.y(),
               _point_cloud_viewport.width(), _point_cloud_viewport.height());

    // CAVEAT: re-express the point cloud in OpenGL axis convention.
    auto from_cam_to_gl = Eigen::Transform<float, 3, Eigen::Projective>{};
    from_cam_to_gl.setIdentity();
    // clang-format off
    from_cam_to_gl.matrix().topLeftCorner<3, 3>() <<
      1,  0,  0,
      0, -1,  0,
      0,  0, -1;
    // clang-format on

    // Render the point cloud.
    const Eigen::Matrix4f model_view_final = _model_view * _scale_mat;
    _point_cloud_renderer.render(_point_cloud, _point_size,
                                 from_cam_to_gl.matrix(),  //
                                 model_view_final, _point_cloud_projection);
  }

  auto get_framebuffer_sizes() const -> Eigen::Vector2i
  {
    auto sizes = Eigen::Vector2i{};
    glfwGetFramebufferSize(_window, &sizes.x(), &sizes.y());
    return sizes;
  }

private:
  static auto get_self(GLFWwindow* const window) -> SingleWindowApp&
  {
    const auto app_void_ptr = glfwGetWindowUserPointer(window);
    if (app_void_ptr == nullptr)
      throw std::runtime_error{
          "Please call glfwSetWindowUserPointer to register this window!"};
    const auto app_ptr = reinterpret_cast<SingleWindowApp*>(app_void_ptr);
    return *app_ptr;
  }

  static auto window_size_callback(GLFWwindow* window, const int, const int)
      -> void
  {
    auto& self = get_self(window);

    auto& fb_sizes = self._fb_sizes;
    fb_sizes = self.get_framebuffer_sizes();

    // Point cloud viewport rectangle geometry.
    self._point_cloud_viewport.top_left().setZero();
    self._point_cloud_viewport.sizes() << fb_sizes.x() / 2, fb_sizes.y();

    // Video viewport rectangle geometry.
    self._video_viewport.top_left() << fb_sizes.x() / 2, 0;
    self._video_viewport.sizes() << fb_sizes.x() / 2, fb_sizes.y();

    // Update the current projection matrices.
    auto scale = 0.5f;
    if (self._video_viewport.width() < self._pipeline._video_streamer.width())
      scale *= static_cast<float>(self._pipeline._video_streamer.width()) /
               self._video_viewport.width();
    self._projection = self._video_viewport.orthographic_projection(scale);

    // Point cloud projection matrix.
    self._point_cloud_projection = self._point_cloud_viewport.perspective();
  }

  static auto key_callback(GLFWwindow* window,  //
                           int key,             //
                           int /* scancode */,  //
                           int action,          //
                           int /* mods */) -> void
  {
    auto& app = get_self(window);
    if (app._pause && key == GLFW_KEY_SPACE &&
        (action == GLFW_RELEASE || action == GLFW_REPEAT))
    {
      app._pause = false;
      std::cout << "RESUME" << std::endl;
      return;
    }

    // Use the escape key to smoothly exit the OpenGL app.
    if ((action == GLFW_RELEASE || action == GLFW_REPEAT) &&
        key == GLFW_KEY_ESCAPE)
    {
      app._quit = true;
      return;
    }

    if (action == GLFW_RELEASE || action == GLFW_REPEAT)
    {
      app.move_point_cloud_camera_with_keyboard(key);
      app.resize_point_size(key);
      app.change_camera_step_size(key);
      app.change_model_scale(key);
      return;
    }
  }

  auto move_point_cloud_camera_with_keyboard(const int key) -> void
  {
    if (key == GLFW_KEY_W)
      _point_cloud_camera.move_forward(_delta);
    if (key == GLFW_KEY_S)
      _point_cloud_camera.move_backward(_delta);
    if (key == GLFW_KEY_A)
      _point_cloud_camera.move_left(_delta);
    if (key == GLFW_KEY_D)
      _point_cloud_camera.move_right(_delta);

    if (key == GLFW_KEY_H)
      _point_cloud_camera.no_head_movement(-_angle_delta);  // CCW
    if (key == GLFW_KEY_K)
      _point_cloud_camera.no_head_movement(+_angle_delta);  // CW

    if (key == GLFW_KEY_U)
      _point_cloud_camera.yes_head_movement(+_angle_delta);
    if (key == GLFW_KEY_J)
      _point_cloud_camera.yes_head_movement(-_angle_delta);

    if (key == GLFW_KEY_R)
      _point_cloud_camera.move_up(_delta);
    if (key == GLFW_KEY_F)
      _point_cloud_camera.move_down(_delta);

    if (key == GLFW_KEY_Y)
      _point_cloud_camera.maybe_head_movement(-_angle_delta);
    if (key == GLFW_KEY_I)
      _point_cloud_camera.maybe_head_movement(+_angle_delta);

    _point_cloud_camera.update();
    _model_view = _point_cloud_camera.view_matrix();
  }

  auto resize_point_size(const int key) -> void
  {
    if (key == GLFW_KEY_MINUS)
      _point_size /= 1.1f;

    if (key == GLFW_KEY_EQUAL)
      _point_size *= 1.1f;
  }

  auto change_camera_step_size(const int key) -> void
  {
    if (key == GLFW_KEY_1)
      _delta /= 1.1f;

    if (key == GLFW_KEY_2)
      _delta *= 1.1f;
  }

  auto change_model_scale(const int key) -> void
  {
    if (key == GLFW_KEY_Z)
      _scale_mat.topLeftCorner<3, 3>() /= 1.1f;

    if (key == GLFW_KEY_X)
      _scale_mat.topLeftCorner<3, 3>() *= 1.1f;
  }

private:
  static auto init_glfw() -> void
  {
    if (_glfw_initialized)
      throw std::runtime_error{
          "Error: cannot instantiate more than one GLFW application!"};

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
    {
      const auto err_str =
          reinterpret_cast<const char*>(glewGetErrorString(err));
      throw std::runtime_error{fmt::format(
          "Error: failed to initialize GLEW: {}", std::string_view{err_str})};
    }
#endif
  }

private:
  static bool _glfw_initialized;

  GLFWwindow* _window = nullptr;
  //! @brief Framebuffer sizes
  //! We want to use this and not the window sizes because of MacOS retina
  //! display.
  Eigen::Vector2i _fb_sizes = -Eigen::Vector2i::Ones();

  sara::OdometryPipeline _pipeline;

  //! Video rendering
  //!
  //! The viewport
  k::Viewport _video_viewport;
  //! @brief the video texture.
  kgl::TexturedImage2D _texture;
  //! @brief the video quad.
  kgl::TexturedQuad _quad;
  //! @brief Texture renderer.
  kgl::TextureRenderer _texture_renderer;
  //! @brief Model-view-projection matrices.
  Eigen::Matrix4f _projection;

  //! Point cloud rendering
  //!
  //! @brief The viewport.
  k::Viewport _point_cloud_viewport;
  //! @brief Point cloud GPU data.
  kgl::ColoredPointCloud _point_cloud;
  //! @brief Point cloud GPU renderer.
  kgl::ColoredPointCloudRenderer _point_cloud_renderer;
  //! @brief Point cloud rendering options.
  Eigen::Matrix4f _point_cloud_projection;
  //! @brief Camera of the point cloud scene.
  k::Camera _point_cloud_camera;
  Eigen::Matrix4f _model_view = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f _scale_mat = Eigen::Matrix4f::Identity();
  float _point_size = 1.5f;
  double _delta = (5._m).value;
  double _angle_delta = (10._deg).value;

  //! @brief User interaction.
  bool _pause = false;
  bool _quit = false;
};

bool SingleWindowApp::_glfw_initialized = false;


auto main([[maybe_unused]] int const argc, [[maybe_unused]] char** const argv)
    -> int
{
#if defined(_OPENMP)
  const auto num_threads = omp_get_max_threads();
  omp_set_num_threads(num_threads);
  Eigen::setNbThreads(num_threads);
#endif

#define USE_HARDCODED_VIDEO_PATH
#if defined(USE_HARDCODED_VIDEO_PATH) && defined(__APPLE__)
  const auto video_path =
      fs::path{"/Users/oddkiva/Desktop/datasets/sample-1.mp4"};
  if (!fs::exists(video_path))
  {
    fmt::print("Video {} does not exist", video_path.string());
    return EXIT_FAILURE;
  }
#else
  if (argc < 2)
  {
    std::cout << fmt::format("Usage: {} VIDEO_PATH\n",
                             std::string_view{argv[0]});
    return EXIT_FAILURE;
  }

  const auto video_path = fs::path{argv[1]};
#endif
  auto camera = sara::v2::BrownConradyDistortionModel<double>{};
  {
    camera.fx() = 917.2878392016245;
    camera.fy() = 917.2878392016245;
    camera.shear() = 0.;
    camera.u0() = 960.;
    camera.v0() = 540.;
    // clang-format off
    camera.k() <<
      -0.2338367557617234,
      +0.05952465745165465,
      -0.007947847982157091;
    // clang-format on
    camera.p() << -0.0003137658969742134, 0.00021943576376532096;
  }

  try
  {
    auto app = SingleWindowApp{{800, 600}, "Odometry: " + video_path.string()};
    app.set_config(video_path, camera);
    app.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
