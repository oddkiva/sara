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

#include "Geometry.hpp"
#include "ImagePlaneRenderer.hpp"
#include "MetricGridRenderer.hpp"

#include "MyGLFW.hpp"

#include <drafts/OpenCL/GL.hpp>

#include <DO/Sara/Core/Math/Rotation.hpp>
#include <DO/Sara/ImageIO.hpp>

#ifdef __EMSCRIPTEN__
#  include <emscripten/emscripten.h>
#  define GLFW_INCLUDE_ES3
#  include <filesystem>
#else
#  include <boost/filesystem.hpp>
#endif

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


#ifdef __EMSCRIPTEN__
namespace fs = std::filesystem;
#else
namespace fs = boost::filesystem;
#endif
namespace sara = DO::Sara;


#ifndef __EMSCRIPTEN__
static auto program_dir_path = fs::path{};
#endif

// Extrinsic camera parameter state.
static auto ypr_deg = std::array<float, 3>{0, 0, 0};
static auto rotation_changed = false;

auto update_rotation()
{
  if (!rotation_changed)
    return;

  // clang-format off
  static const Eigen::Matrix3f P = (Eigen::Matrix3f{} <<
     0,  0, 1, // Camera Z =          Automotive X
    -1,  0, 0, // Camera X = Negative Automotive Y
     0, -1, 0  // Camera Y = Negative Automotive Z
  ).finished();
  // clang-format on
  const Eigen::Matrix3f R =
      sara::rotation(ypr_deg[0] * static_cast<float>(M_PI) / 180.f,
                     ypr_deg[1] * static_cast<float>(M_PI) / 180.f,
                     ypr_deg[2] * static_cast<float>(M_PI) / 180.f) *
      P;
  const auto t = Eigen::Vector3f(0, 0, 1.51);

  auto& grid_renderer = MetricGridRenderer::instance();
  auto& line_batches = grid_renderer._lines;
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
    angles.innerHTML += "<li>YAW = " + $0 + "</li>";
    angles.innerHTML += "<li>PITCH = " + $1 + "</li>";
    angles.innerHTML += "<li>ROLL = " + $2 + "</li>";
    angles.innerHTML += "</ul>";
  }, ypr_deg[0], ypr_deg[1], ypr_deg[2]);
  // clang-format on
#endif

  rotation_changed = false;
}

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

  auto& image = ImagePlaneRenderer::instance()._textures.front();

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
    ypr_deg[0] += angle_step;
    rotation_changed = true;
    break;
  case GLFW_KEY_D:
    ypr_deg[0] -= angle_step;
    rotation_changed = true;
    break;
  case GLFW_KEY_W:
    ypr_deg[1] += angle_step;
    rotation_changed = true;
    break;
  case GLFW_KEY_S:
    ypr_deg[1] -= angle_step;
    rotation_changed = true;
    break;
  case GLFW_KEY_Q:
    ypr_deg[2] += angle_step;
    rotation_changed = true;
    break;
  case GLFW_KEY_E:
    ypr_deg[2] -= angle_step;
    rotation_changed = true;
    break;
  default:
    break;
  };

  update_rotation();
}

void scroll_callback(GLFWwindow* /*window*/, double /*xoffset */,
                     double yoffset)
{
  auto& image = ImagePlaneRenderer::instance()._textures.front();

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

void override_callbacks()
{
  // Set the appropriate mouse and keyboard callbacks.
  glfwSetWindowSizeCallback(MyGLFW::window, window_size_callback);
  glfwSetKeyCallback(MyGLFW::window, key_callback);
  glfwSetScrollCallback(MyGLFW::window, scroll_callback);
}


auto render_frame() -> void
{
  glfwPollEvents();

  // Start the Dear ImGui frame
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  // Render your GUI
  {
    const auto ypr_old = ypr_deg;
    ImGui::Begin("Camera Orientation");
    ImGui::SliderFloat("Yaw", &ypr_deg[0], -180.f, 180.f);
    ImGui::SliderFloat("Pitch", &ypr_deg[1], -90.f, 90.f);
    ImGui::SliderFloat("Roll", &ypr_deg[2], -180.f, 180.f);
    ImGui::End();

    rotation_changed = !std::equal(ypr_old.begin(), ypr_old.end(),  //
                                   ypr_deg.begin());
    update_rotation();
  }
  ImGui::Render();


  // Clear the screen.
  glViewport(0, 0, MyGLFW::width, MyGLFW::height);
  glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Render the scene.
  {
    auto& image_plane_renderer = ImagePlaneRenderer::instance();
    const auto& image_texture = image_plane_renderer._textures.front();
    image_plane_renderer.render(image_texture);

    auto& grid_renderer = MetricGridRenderer::instance();
    const auto& lines = grid_renderer._lines;
    for (auto i = 0u; i < lines.size(); ++i)
      grid_renderer.render(image_texture, lines[i]);
  }

  // Render ImGUI.
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  glfwSwapBuffers(MyGLFW::window);
}


auto initialize_image_texture()
{
#ifdef __EMSCRIPTEN__
  const auto image = sara::imread<sara::Rgb8>("assets/image-omni.png");
#else
  const auto image = sara::imread<sara::Rgb8>(
      (program_dir_path / "assets/image-omni.png").string());
#endif

  auto& image_plane_renderer = ImagePlaneRenderer::instance();
  auto& image_textures = image_plane_renderer._textures;
  image_textures.resize(1);

  // Transfer the CPU image data to the GPU texture.
  static constexpr auto texture_unit = 0;
  auto& image_texture = image_plane_renderer._textures.front();
  image_texture.set_texture(image, texture_unit);

  // Geometry
  const auto aspect_ratio = static_cast<float>(MyGLFW::width) / MyGLFW::height;
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

auto initialize_metric_grid(const std::pair<std::int32_t, std::int32_t>& xrange,
                            const std::pair<std::int32_t, int32_t>& yrange,
                            float square_size_in_meters = 1.f,
                            float line_discretization_step = 0.25f) -> void
{
  const auto& sq_size = square_size_in_meters;
  const auto& s = line_discretization_step;

  auto& grid_renderer = MetricGridRenderer::instance();
  auto& gl_lines = grid_renderer._lines;
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
  auto& image_plane_renderer = ImagePlaneRenderer::instance();
  image_plane_renderer.destroy_gl_objects();

  auto& grid_renderer = MetricGridRenderer::instance();
  grid_renderer.destroy_gl_objects();

  // Destroy the image textures.
  auto& image_textures = image_plane_renderer._textures;
  for (auto i = 0u; i < image_textures.size(); ++i)
    image_textures[i].destroy();
  image_textures.clear();
  image_textures.clear();

  // Destroy the line buffers.
  auto& lines = grid_renderer._lines;
  for (auto i = 0u; i < image_textures.size(); ++i)
    lines[i].destroy();
  lines.clear();
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
    glewInit();  // Otherwise shaders won't work and the program crashes.
#endif

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    // ImGuiIO &io = ImGui::GetIO();
    ImGui_ImplGlfw_InitForOpenGL(MyGLFW::window, true);
    ImGui_ImplOpenGL3_Init(MyGLFW::glsl_version.c_str());
    ImGui::StyleColorsDark();

    auto& image_plane_renderer = ImagePlaneRenderer::instance();
    image_plane_renderer.initialize();
    initialize_image_texture();

    auto& metric_grid_renderer = MetricGridRenderer::instance();
    metric_grid_renderer.initialize();
    initialize_metric_grid({-50, 50}, {-50, 50});

    // We are ready to override the callbacks.
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

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
