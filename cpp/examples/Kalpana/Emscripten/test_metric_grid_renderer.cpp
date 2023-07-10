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

#include <DO/Kalpana/EasyGL.hpp>

#include <DO/Sara/Core/Math/Rotation.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>

#include <GLFW/glfw3.h>

#ifdef __EMSCRIPTEN__
#  include <emscripten/emscripten.h>
#  define GLFW_INCLUDE_ES3
#endif

#include <filesystem>

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "Geometry.hpp"
#include "ImagePlaneRenderer.hpp"
#include "MetricGridRenderer.hpp"


namespace fs = std::filesystem;
namespace sara = DO::Sara;


class GLFWApp
{
public:
  GLFWApp(const Eigen::Vector2i& sizes,
          const std::string& title = "Image Dewarper")
    : _window_sizes{sizes}
  {
    // In the constructor, the order in which functions are called matters.
    //
    // 1. Init GLFW.
    if (glfwInit() != GL_TRUE)
      throw std::runtime_error{"Failed to initialize GLFW!"};

    // 2. Specify the OpenGL version we want to use.
    //    OpenGL core version 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // 3. Create a GLFW window.
    _window = glfwCreateWindow(sizes.x(), sizes.y(),  //
                               title.c_str(),         //
                               nullptr, nullptr);
    if (!_window)
    {
      glfwTerminate();
      throw std::runtime_error{"Failed to create window!"};
    }

    // clang-format off
// #ifdef __APPLE__
//   // GL 3.2 + GLSL 150
//   MyGLFW::glsl_version = "#version 150";
//   glfwWindowHint(  // required on Mac OS
//       GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
// #elif __linux__
//   // GL 3.2 + GLSL 150
//   glsl_version = "#version 150";
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
// #elif _WIN32
//   // GL 3.0 + GLSL 130
//   glsl_version = "#version 130";
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
// #endif
    // clang-format on

#ifdef _WIN32
    // if it's a HighDPI monitor, try to scale everything
    const auto monitor = glfwGetPrimaryMonitor();
    float xscale, yscale;
    glfwGetMonitorContentScale(monitor, &xscale, &yscale);
    if (xscale > 1 || yscale > 1)
    {
      high_dpi_scale_factor = static_cast<int>(xscale);
      glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    }
#elif __APPLE__
    // to prevent 1200x800 from becoming 2400x1600
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif

    // 4. Tell that the OpenGL rendering will be done on this window surface.
    glfwMakeContextCurrent(_window);

#ifndef __EMSCRIPTEN__
    // 5. Load GLEW.
    //    This is important as any attempt at calling OpenGL functions will
    //    result in a runtime crash.
    const auto err = glewInit();
    if (err != GLEW_OK)
      throw std::runtime_error{"Error: failed to initialize GLEW!"};
#endif

    // 6. Register callbacks to GLFW.
    glfwSetWindowUserPointer(_window, this);
    glfwSetWindowSizeCallback(_window, window_size_callback);
    glfwSetKeyCallback(_window, key_callback);
    glfwSetScrollCallback(_window, scroll_callback);
  }

  auto initialize(const fs::path& program_dir_path) -> void
  {
#ifndef __EMSCRIPTEN__
    _program_dir_path = program_dir_path;
#endif

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    // ImGuiIO &io = ImGui::GetIO();
    ImGui_ImplGlfw_InitForOpenGL(_window, true);
    ImGui_ImplOpenGL3_Init("#version 300 es");
    ImGui::StyleColorsDark();

    _image_plane_renderer.initialize();
    initialize_image_texture();

    _metric_grid_renderer.initialize();
    initialize_metric_grid({-50, 50}, {-50, 50});

    // Specific rendering options.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
  }

  auto run() -> void
  {
#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(render_frame, 0, 1);
#else
    while (!glfwWindowShouldClose(_window))
      render_frame();
#endif
  }

  auto terminate() -> void
  {
    cleanup_gl_objects();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
  }

  auto update_rotation() -> void
  {
    if (!_rotation_changed)
      return;

    // clang-format off
    static const Eigen::Matrix3f P = (Eigen::Matrix3f{} <<
       0,  0, 1, // Camera Z =          Automotive X
      -1,  0, 0, // Camera X = Negative Automotive Y
       0, -1, 0  // Camera Y = Negative Automotive Z
    ).finished();
    // clang-format on
    const Eigen::Matrix3f R =
        sara::rotation(  //
            _ypr_deg[0] * static_cast<float>(M_PI) / 180.f,
            _ypr_deg[1] * static_cast<float>(M_PI) / 180.f,
            _ypr_deg[2] * static_cast<float>(M_PI) / 180.f) *
        P;
    const auto t = Eigen::Vector3f(0, 0, 1.51f);

    auto& line_batches = _metric_grid_renderer._lines;
    for (auto& lines : line_batches)
    {
      lines._extrinsics.topLeftCorner(3, 3) = R.transpose();
      lines._extrinsics.block<3, 1>(0, 3) = -R.transpose() * t;
    }

#ifdef __EMSCRIPTEN__
    // clang-format off
    EM_ASM({
      var angles = document.getElementById('angles');
      angles.innerHTML = "<ul>";
      angles.innerHTML += "  <li>YAW = " + $0 + "</li>";
      angles.innerHTML += "  <li>PITCH = " + $1 + "</li>";
      angles.innerHTML += "  <li>ROLL = " + $2 + "</li>";
      angles.innerHTML += "</ul>";
    }, _ypr_deg[0], _ypr_deg[1], _ypr_deg[2]);
    // clang-format on
#endif

    _rotation_changed = false;
  }

private:
  auto render_frame() -> void
  {
    glfwPollEvents();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    // Render your GUI
    {
      const auto ypr_old = _ypr_deg;
      ImGui::Begin("Camera Orientation");
      ImGui::SliderFloat("Yaw", &_ypr_deg[0], -180.f, 180.f);
      ImGui::SliderFloat("Pitch", &_ypr_deg[1], -90.f, 90.f);
      ImGui::SliderFloat("Roll", &_ypr_deg[2], -180.f, 180.f);
      ImGui::End();

      _rotation_changed = !std::equal(ypr_old.begin(), ypr_old.end(),  //
                                      _ypr_deg.begin());
      update_rotation();
    }
    ImGui::Render();


    // Clear the screen.
    glViewport(0, 0, _window_sizes.x(), _window_sizes.y());
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render the scene.
    {
      const auto& image_texture = _image_plane_renderer._textures.front();
      _image_plane_renderer.render(image_texture);

      const auto& lines = _metric_grid_renderer._lines;
      for (auto i = 0u; i < lines.size(); ++i)
        _metric_grid_renderer.render(image_texture, lines[i]);
    }

    // Render ImGUI.
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(_window);
  }

  auto initialize_image_texture() -> void
  {
#ifdef __EMSCRIPTEN__
    const auto image = sara::imread<sara::Rgb8>("assets/image-omni.png");
#elif defined _WIN32
    const auto image = sara::resize(
        sara::imread<sara::Rgb8>(
            "C:/Users/David/Desktop/GitLab/sara/data/stinkbug.png"),
        {1920, 1080});
#else
    const auto image = sara::imread<sara::Rgb8>(
        (_program_dir_path / "assets/image-omni.png").string());
#endif

    auto& image_textures = _image_plane_renderer._textures;
    image_textures.resize(1);

    // Transfer the CPU image data to the GPU texture.
    static constexpr auto texture_unit = 0;
    auto& image_texture = _image_plane_renderer._textures.front();
    image_texture.set_texture(image, texture_unit);

    // Geometry
    const auto aspect_ratio = static_cast<float>(_window_sizes.x()) /  //
                              _window_sizes.y();
    image_texture._model_view.setIdentity();
    image_texture._projection = orthographic(
        -0.5f * aspect_ratio, 0.5f * aspect_ratio, -0.5f, 0.5f, -0.5f, 0.5f);
  }

  auto initialize_camera_parameters(MetricGridRenderer::LineShaderData& lines)
      -> void
  {
    // The conversion from the automotive axis convention to the computer vision
    // convention.
    //
    // clang-format off
  const Eigen::Matrix3f P = (Eigen::Matrix3f{} <<
     0,  0, 1,
    -1,  0, 0,
     0, -1, 0
  ).finished();
    // clang-format on

    auto& C = lines._extrinsics;
    C.setIdentity();
    C.topLeftCorner<3, 3>() = P.transpose();
    C.col(3).head(3) = -P.transpose() * Eigen::Vector3f{0.f, 0.f, 1.51f};

    auto& intrinsics = lines._intrinsics;

    // clang-format off
  const auto K = (Eigen::Matrix3f{} <<
    1041.55762f, -2.31719828f, 942.885742f,
            0.f,  1041.53857f, 589.198425f,
            0.f,          0.f,         1.f
  ).finished();
  intrinsics.set_calibration_matrix(K);
  intrinsics.radial_distortion_coefficients <<
     0.442631334f,
    -0.156340882f,
     0;
  intrinsics.tangential_distortion_coefficients <<
    -0.000787709199f,
    -0.000381082471f;
    // clang-format on
    intrinsics.xi = 1.43936455f;
  }

  auto initialize_metric_grid(const std::pair<int, int>& xrange,
                              const std::pair<int, int>& yrange,
                              float square_size_in_meters = 1.f,
                              float line_discretization_step = 0.25f) -> void
  {
    const auto& sq_size = square_size_in_meters;
    const auto& s = line_discretization_step;

    auto& gl_lines = _metric_grid_renderer._lines;
    gl_lines.resize(2);

    auto line_data = MetricGridRenderer::LineHostData{};

    // Draw y-level sets.
    for (auto y = static_cast<float>(yrange.first); y <= yrange.second;
         y += sq_size)
    {
      for (auto x = static_cast<float>(xrange.first); x < xrange.second; x += s)
      {
        const auto a = Eigen::Vector2f(x, y);
        const auto b = Eigen::Vector2f(x + s, y);
        line_data.add_line_segment(a, b, 10.f / 1080, 0.5f / 1080);
      }
    }
    initialize_camera_parameters(gl_lines[0]);
    gl_lines[0].set_data(line_data);
    gl_lines[0]._color << 0.8f, 0.0f, 0.0f, 0.5f;

    line_data.clear();

    // Draw x-level sets.
    for (auto x = static_cast<float>(xrange.first); x <= xrange.second;
         x += sq_size)
    {
      for (auto y = static_cast<float>(yrange.first); y < yrange.second; y += s)
      {
        const auto a = Eigen::Vector2f(x, y);
        const auto b = Eigen::Vector2f(x, y + s);
        line_data.add_line_segment(a, b, 10.f / 1080, 0.5f / 1080);
      }
    }
    initialize_camera_parameters(gl_lines[1]);
    gl_lines[1].set_data(line_data);
    gl_lines[1]._color << 0.0f, 0.0f, 0.8f, 0.5f;
  }

  auto cleanup_gl_objects() -> void
  {
    // Destroy the shaders and quad geometry data.
    _image_plane_renderer.destroy_gl_objects();
    _metric_grid_renderer.destroy_gl_objects();

    // Destroy the image textures.
    auto& image_textures = _image_plane_renderer._textures;
    for (auto i = 0u; i < image_textures.size(); ++i)
      image_textures[i].destroy();
    image_textures.clear();
    image_textures.clear();

    // Destroy the line buffers.
    auto& lines = _metric_grid_renderer._lines;
    for (auto i = 0u; i < image_textures.size(); ++i)
      lines[i].destroy();
    lines.clear();
  }

public: /* callbacks */
  static auto window_size_callback(GLFWwindow* window, int width, int height)
      -> void
  {
    auto& app = get_app(window);
    app._window_sizes << width, height;
    const auto aspect_ratio = static_cast<float>(width) / height;

    auto& image = app._image_plane_renderer._textures.front();
    image._projection = orthographic(-0.5f * aspect_ratio, 0.5f * aspect_ratio,
                                     -0.5f, 0.5f, -0.5f, 0.5f);
  }

  static auto key_callback(GLFWwindow* window, int key, int /* scancode */,
                           int action, int /* modifier */) -> void
  {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
      glfwSetWindowShouldClose(window, 1);

    if (action == GLFW_RELEASE)
      return;

    auto& app = get_app(window);
    auto& image = app._image_plane_renderer._textures.front();

    static constexpr auto angle_step = 0.5f;

    switch (key)
    {
    case GLFW_KEY_LEFT:
      image._model_view(0, 3) += 0.01f;
      break;
    case GLFW_KEY_RIGHT:
      image._model_view(0, 3) -= 0.01f;
      break;
    case GLFW_KEY_UP:
      image._model_view(1, 3) += 0.01f;
      break;
    case GLFW_KEY_DOWN:
      image._model_view(1, 3) -= 0.01f;
      break;
    case GLFW_KEY_A:
      app._ypr_deg[0] += angle_step;
      app._rotation_changed = true;
      break;
    case GLFW_KEY_D:
      app._ypr_deg[0] -= angle_step;
      app._rotation_changed = true;
      break;
    case GLFW_KEY_W:
      app._ypr_deg[1] += angle_step;
      app._rotation_changed = true;
      break;
    case GLFW_KEY_S:
      app._ypr_deg[1] -= angle_step;
      app._rotation_changed = true;
      break;
    case GLFW_KEY_Q:
      app._ypr_deg[2] += angle_step;
      app._rotation_changed = true;
      break;
    case GLFW_KEY_E:
      app._ypr_deg[2] -= angle_step;
      app._rotation_changed = true;
      break;
    default:
      break;
    };

    app.update_rotation();
  }

  static auto scroll_callback(GLFWwindow* window, double /*xoffset */,
                              double yoffset) -> void
  {
    auto& app = get_app(window);
    auto& image = app._image_plane_renderer._textures.front();

    if (yoffset > 0)
    {
      image._model_view.topLeftCorner(3, 4) *= 1.05f;
      return;
    }
    if (yoffset < 0)
    {
      image._model_view.topLeftCorner(3, 4) /= 1.05f;
      return;
    }
  }

private: /* convenience free functions. */
  static auto get_app(GLFWwindow* window) -> GLFWApp&
  {
    const auto app_ptr =
        reinterpret_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
    if (app_ptr == nullptr)
      throw std::runtime_error{
          "Check that the app is registered with glfwSetWindowUserPointer!"};
    return *app_ptr;
  }

private:
  GLFWwindow* _window = nullptr;
  Eigen::Vector2i _window_sizes = Eigen::Vector2i::Zero();

#ifndef __EMSCRIPTEN__
  fs::path _program_dir_path;
#endif

  // Extrinsic camera parameter state.
  std::array<float, 3> _ypr_deg = {0, 0, 0};
  bool _rotation_changed = false;

  ImagePlaneRenderer _image_plane_renderer;
  MetricGridRenderer _metric_grid_renderer;
};

int main(int, [[maybe_unused]] char** argv)
{
  try
  {
    auto app = GLFWApp{{800, 600}};
    // app.initialize(fs::path{argv[0]}.parent_path());
    app.initialize(fs::path{"/home/david/GitLab/DO-CV"} / "sara" / "cpp" /
                   "examples" / "Kalpana" / "Emscripten");
    app.run();
    app.terminate();
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
