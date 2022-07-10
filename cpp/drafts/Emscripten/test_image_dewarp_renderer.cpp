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

#include <DO/Sara/Core/Math/Rotation.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>

#ifdef __EMSCRIPTEN__
#  include <emscripten/emscripten.h>
#  define GLFW_INCLUDE_ES3
#else
#  include <boost/filesystem.hpp>
#endif


#include "MyGLFW.hpp"

#include "Geometry.hpp"
#include "ImageDewarpRenderer.hpp"
#include "ImagePlaneRenderer.hpp"


#ifdef __EMSCRIPTEN__
namespace fs = std::filesystem;
#else
namespace fs = boost::filesystem;
#endif
namespace sara = DO::Sara;


#ifndef __EMSCRIPTEN__
static auto program_dir_path = fs::path{};
#endif
static auto dewarp_mode = 0;
static auto camera_params = ImageDewarpRenderer::CameraParameters{};


auto window_size_callback(GLFWwindow* /* window */, int width, int height)
    -> void
{
  MyGLFW::width = width;
  MyGLFW::height = height;
  const auto aspect_ratio = static_cast<float>(width) / height;

  auto& image = ImagePlaneRenderer::instance()._textures.front();
  image._projection = orthographic(-0.5f * aspect_ratio, 0.5f * aspect_ratio,
                                   -0.5f, 0.5f, -0.5f, 0.5f);
}

auto key_callback(GLFWwindow* /* window */, int key, int /* scancode */,
                  int action, int /* modifier */) -> void
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
    glfwSetWindowShouldClose(MyGLFW::window, 1);

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
    camera_params.K(0, 2) += delta;
    K_changed = true;
    break;
  case GLFW_KEY_RIGHT:
    camera_params.K(0, 2) -= delta;
    K_changed = true;
    break;
  case GLFW_KEY_UP:
    camera_params.K(1, 2) += delta;
    K_changed = true;
    break;
  case GLFW_KEY_DOWN:
    camera_params.K(1, 2) -= delta;
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
    dewarp_mode = (dewarp_mode + 1) % 2;
    break;
  default:
    break;
  };

  if (K_changed)
  {
    camera_params.K_inverse = camera_params.K.inverse();
    K_changed = false;
  }

  if (rotation_changed)
  {
    camera_params.R =
        sara::rotation(yaw_pitch_roll[0], yaw_pitch_roll[1], yaw_pitch_roll[2]);
    rotation_changed = false;
  }
}

void scroll_callback(GLFWwindow* /*window*/, double /*xoffset */,
                     double yoffset)
{
  if (yoffset > 0)
  {
    camera_params.K(0, 0) *= 1.05f;
    camera_params.K(1, 1) *= 1.05f;
    camera_params.K_inverse = camera_params.K.inverse();
    return;
  }
  if (yoffset < 0)
  {
    camera_params.K(0, 0) /= 1.05f;
    camera_params.K(1, 1) /= 1.05f;
    camera_params.K_inverse = camera_params.K.inverse();
    return;
  }
}

void override_callbacks()
{
  // Set the appropriate mouse and keyboard callbacks.
  glfwSetWindowSizeCallback(MyGLFW::window, window_size_callback);
  glfwSetKeyCallback(MyGLFW::window, key_callback);
  glfwSetScrollCallback(MyGLFW::window, scroll_callback);
}


auto render_frame() -> void
{
  glViewport(0, 0, MyGLFW::width, MyGLFW::height);

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  const auto& image_plane_renderer = ImagePlaneRenderer::instance();
  const auto& image_texture = image_plane_renderer._textures.front();

  auto& image_dewarp_renderer = ImageDewarpRenderer::instance();
  image_dewarp_renderer.render(image_texture, camera_params, dewarp_mode);

  glfwSwapBuffers(MyGLFW::window);
  glfwPollEvents();
}

auto initialize_image_textures()
{
#ifdef __EMSCRIPTEN__
  // const auto image = sara::imread<sara::Rgb8>("assets/image-omni.png");
  auto image = sara::Image<sara::Rgb8>{1920, 1080};
  image.flat_array().fill(sara::White8);
#elif defined _WIN32
  const auto image =
      sara::resize(sara::imread<sara::Rgb8>(
                       "C:/Users/David/Desktop/GitLab/sara/data/stinkbug.png"),
                   {1920, 1080});
#else
  const auto image = sara::imread<sara::Rgb8>(
      (program_dir_path / "assets/image-omni.png").string());
#endif

  auto& image_plane_renderer = ImagePlaneRenderer::instance();
  auto& image_textures = image_plane_renderer._textures;
  image_textures.resize(1);

  auto& image_texture = image_textures.front();
  // Transfer the CPU image data to the GPU texture.
  static constexpr auto texture_unit = 0;
  image_texture.set_texture(image, texture_unit);

  // Geometry
  image_texture._model_view.setIdentity();
  const auto aspect_ratio = static_cast<float>(MyGLFW::width) / MyGLFW::height;
  image_texture._projection =
      orthographic(-0.5f * aspect_ratio, 0.5f * aspect_ratio,  //
                   -0.5f, 0.5f,                                //
                   -0.5f, 0.5f);
}

auto initialize_camera_parameters() -> void
{
  // clang-format off
  const auto K = (Eigen::Matrix3f{} <<
    1041.55762f, -2.31719828f, 942.885742f,
            0.f,  1041.53857f, 589.198425f,
            0.f,          0.f,         1.f
  ).finished();
  camera_params._intrinsics.set_calibration_matrix(K);
  camera_params._intrinsics.radial_distortion_coefficients <<
     0.442631334f,
    -0.156340882f,
     0;
  camera_params._intrinsics.tangential_distortion_coefficients <<
    -0.000787709199f,
    -0.000381082471f;
  // clang-format on
  camera_params._intrinsics.xi = 1.43936455f;

  // Destination stereographic reprojection.
  camera_params.R.setIdentity();
  camera_params.K = K;
  camera_params.K_inverse = K.inverse();
}


auto cleanup_gl_objects() -> void
{
  // Destroy the shaders and quad geometry data.
  auto& image_plane_renderer = ImagePlaneRenderer::instance();
  image_plane_renderer.destroy_gl_objects();

  // Destroy the image textures.
  auto& image_textures = image_plane_renderer._textures;
  for (auto i = 0u; i < image_textures.size(); ++i)
    image_textures[i].destroy();
  image_textures.clear();

  auto& image_dewarp_renderer = ImageDewarpRenderer::instance();
  image_dewarp_renderer.destroy_gl_objects();
}

int main(int, [[maybe_unused]] char** argv)
{
  try
  {
#ifndef __EMSCRIPTEN__
    program_dir_path = fs::path{argv[0]}.parent_path();
#endif

    if (!MyGLFW::initialize())
      return EXIT_FAILURE;

#ifndef __EMSCRIPTEN__
    glewInit();
#endif

    auto& image_plane_renderer = ImagePlaneRenderer::instance();
    image_plane_renderer.initialize();
    initialize_image_textures();

    auto& image_dewarp_renderer = ImageDewarpRenderer::instance();
    image_dewarp_renderer.initialize();
    initialize_camera_parameters();

    override_callbacks();

    // Specific rendering options.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop(render_frame, 0, 1);
#else
    while (!glfwWindowShouldClose(MyGLFW::window))
      render_frame();
#endif

    cleanup_gl_objects();

    glfwTerminate();
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
