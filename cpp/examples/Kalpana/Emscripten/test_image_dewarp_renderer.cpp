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

#include "Geometry.hpp"
#include "ImageDewarpRenderer.hpp"
#include "ImagePlaneRenderer.hpp"


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

    initialize_image_textures();
    initialize_camera_parameters();

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
    glfwTerminate();
  }

private:
  auto initialize_image_textures() -> void
  {
    _image_plane_renderer.initialize();

#ifdef __EMSCRIPTEN__
    // const auto image = sara::imread<sara::Rgb8>("assets/image-omni.png");
    auto image = sara::Image<sara::Rgb8>{1920, 1080};
    image.flat_array().fill(sara::White8);
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

    auto& image_texture = image_textures.front();
    // Transfer the CPU image data to the GPU texture.
    static constexpr auto texture_unit = 0;
    image_texture.set_texture(image, texture_unit);

    // Geometry
    image_texture._model_view.setIdentity();
    const auto aspect_ratio =
        static_cast<float>(_window_sizes.x()) / _window_sizes.y();
    image_texture._projection =
        orthographic(-0.5f * aspect_ratio, 0.5f * aspect_ratio,  //
                     -0.5f, 0.5f,                                //
                     -0.5f, 0.5f);
  }

  auto initialize_camera_parameters() -> void
  {
    _image_dewarp_renderer.initialize();

    // clang-format off
    const auto K = (Eigen::Matrix3f{} <<
      1041.55762f, -2.31719828f, 942.885742f,
              0.f,  1041.53857f, 589.198425f,
              0.f,          0.f,         1.f
    ).finished();
    _camera_params._intrinsics.set_calibration_matrix(K);
    _camera_params._intrinsics.radial_distortion_coefficients <<
       0.442631334f,
      -0.156340882f,
       0;
    _camera_params._intrinsics.tangential_distortion_coefficients <<
      -0.000787709199f,
      -0.000381082471f;
    // clang-format on
    _camera_params._intrinsics.xi = 1.43936455f;

    // Destination stereographic reprojection.
    _camera_params.R.setIdentity();
    _camera_params.K = K;
    _camera_params.K_inverse = K.inverse();
  }

  auto render_frame() -> void
  {
    glViewport(0, 0, _window_sizes.x(), _window_sizes.y());

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const auto& image_texture = _image_plane_renderer._textures.front();
    _image_dewarp_renderer.render(image_texture, _camera_params, _dewarp_mode);

    glfwSwapBuffers(_window);
    glfwPollEvents();
  }

  auto cleanup_gl_objects() -> void
  {
    // Destroy the shaders and quad geometry data.
    _image_plane_renderer.destroy_gl_objects();

    // Destroy the image textures.
    auto& image_textures = _image_plane_renderer._textures;
    for (auto i = 0u; i < image_textures.size(); ++i)
      image_textures[i].destroy();
    image_textures.clear();

    _image_dewarp_renderer.destroy_gl_objects();
  }

private: /* callbacks */
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
    auto& app = get_app(window);
    if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
      glfwSetWindowShouldClose(window, 1);

    if (action == GLFW_RELEASE)
      return;

    static auto yaw_pitch_roll = std::array<float, 3>{0, 0, 0};
    static auto K_changed = false;
    static auto rotation_changed = false;

    static constexpr auto angle_step = 0.5f * static_cast<float>(M_PI) / 180;
    static constexpr auto delta = 10.f;

    switch (key)
    {
    case GLFW_KEY_LEFT:
      app._camera_params.K(0, 2) += delta;
      K_changed = true;
      break;
    case GLFW_KEY_RIGHT:
      app._camera_params.K(0, 2) -= delta;
      K_changed = true;
      break;
    case GLFW_KEY_UP:
      app._camera_params.K(1, 2) += delta;
      K_changed = true;
      break;
    case GLFW_KEY_DOWN:
      app._camera_params.K(1, 2) -= delta;
      K_changed = true;
      break;
    case GLFW_KEY_A:
      yaw_pitch_roll[0] += angle_step;
      rotation_changed = true;
      break;
    case GLFW_KEY_D:
      yaw_pitch_roll[0] -= angle_step;
      rotation_changed = true;
      break;
    case GLFW_KEY_W:
      yaw_pitch_roll[2] += angle_step;
      rotation_changed = true;
      break;
    case GLFW_KEY_S:
      yaw_pitch_roll[2] -= angle_step;
      rotation_changed = true;
      break;
    case GLFW_KEY_Q:
      yaw_pitch_roll[1] += angle_step;
      rotation_changed = true;
      break;
    case GLFW_KEY_E:
      yaw_pitch_roll[1] -= angle_step;
      rotation_changed = true;
      break;
    case GLFW_KEY_SPACE:
      app._dewarp_mode = (app._dewarp_mode + 1) % 2;
      break;
    default:
      break;
    };

    if (K_changed)
    {
      app._camera_params.K_inverse = app._camera_params.K.inverse();
      K_changed = false;
    }

    if (rotation_changed)
    {
      app._camera_params.R = sara::rotation(yaw_pitch_roll[0],  //
                                            yaw_pitch_roll[1],  //
                                            yaw_pitch_roll[2]);
      rotation_changed = false;
    }
  }

  static auto scroll_callback(GLFWwindow* window, double /*xoffset */,
                              double yoffset) -> void
  {
    auto& app = get_app(window);
    if (yoffset > 0)
    {
      app._camera_params.K(0, 0) *= 1.05f;
      app._camera_params.K(1, 1) *= 1.05f;
      app._camera_params.K_inverse = app._camera_params.K.inverse();
      return;
    }
    if (yoffset < 0)
    {
      app._camera_params.K(0, 0) /= 1.05f;
      app._camera_params.K(1, 1) /= 1.05f;
      app._camera_params.K_inverse = app._camera_params.K.inverse();
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

  ImagePlaneRenderer _image_plane_renderer;
  ImageDewarpRenderer _image_dewarp_renderer;

#ifndef __EMSCRIPTEN__
  fs::path _program_dir_path;
#endif
  int _dewarp_mode = 0;
  ImageDewarpRenderer::CameraParameters _camera_params;
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
