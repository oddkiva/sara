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

#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>

#include <GLFW/glfw3.h>

#if defined(__EMSCRIPTEN__)
#  include <emscripten/emscripten.h>
#  define GLFW_INCLUDE_ES3
#endif

#include <filesystem>

#include "ImagePlaneRenderer.hpp"
#include "LineRenderer.hpp"


namespace fs = std::filesystem;
namespace sara = DO::Sara;


class GLFWApp;

#if defined(__EMSCRIPTEN__)
auto render_frame_for_emscripten() -> void;
#endif


class GLFWApp
{
  GLFWApp(const Eigen::Vector2i& sizes, const std::string& title)
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
#if defined(__APPLE__)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    // Retina display.
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif

      // clang-format off
// #if defined(__APPLE__)
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

#if defined(_WIN32)
    // if it's a HighDPI monitor, try to scale everything
    const auto monitor = glfwGetPrimaryMonitor();
    float xscale, yscale;
    glfwGetMonitorContentScale(monitor, &xscale, &yscale);
    if (xscale > 1 || yscale > 1)
    {
      _high_dpi_scale_factor = static_cast<int>(xscale);
      glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    }
#elif defined(__APPLE__)
    // to prevent 1200x800 from becoming 2400x1600
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif

    // 3. Create a GLFW window.
    _window = glfwCreateWindow(sizes.x(), sizes.y(),  //
                               title.c_str(),         //
                               nullptr, nullptr);
    if (!_window)
      throw std::runtime_error{"Failed to create window!"};

    // 4. Tell that the OpenGL rendering will be done on this window surface.
    glfwMakeContextCurrent(_window);

#if !defined(__EMSCRIPTEN__) && !defined(__APPLE__)
    // 5. Load GLEW.
    //    This is important as any attempt at calling OpenGL functions will
    //    result in a runtime crash.
    const auto err = glewInit();
    if (err != GLEW_OK)
      throw std::runtime_error{"Error: failed to initialize GLEW!"};
#endif
  }

public:
  ~GLFWApp()
  {
    terminate();
  }

  static auto instantiate(const Eigen::Vector2i& sizes,
                          const std::string& title) -> GLFWApp&
  {
    _instance.reset(new GLFWApp{sizes, title});
    return *_instance;
  }

  static auto instance() -> GLFWApp&
  {
    if (_instance.get() == nullptr)
      throw std::runtime_error{
          "Error: you must instantiate the app exlicitly!"};
    return *_instance;
  }

  auto initialize([[maybe_unused]] const fs::path& program_dir_path) -> void
  {
#if !defined(__EMSCRIPTEN__)
    _program_dir_path = program_dir_path;
#endif

    initialize_image_textures();

    _line_renderer.initialize();
    initialize_lines();

    // Specific rendering options.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // glEnable(GL_DEPTH_TEST);
  }

  auto run() -> void
  {
#if defined(__EMSCRIPTEN__)
    emscripten_set_main_loop(render_frame_for_emscripten, 0, 1);
#else
    while (!glfwWindowShouldClose(_window))
      render_frame();
#endif
  }

  auto render_frame() -> void
  {
    glViewport(0, 0, _window_sizes.x(), _window_sizes.y());

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const auto& image_textures = _image_plane_renderer._textures;
    const auto& lines = _line_renderer._lines;

    for (auto i = 0u; i < image_textures.size(); ++i)
      _image_plane_renderer.render(image_textures[i]);

    for (auto i = 0u; i < image_textures.size(); ++i)
      _line_renderer.render(image_textures[i], lines[i]);

    glfwSwapBuffers(_window);
    glfwPollEvents();
  }

private:
  auto initialize_image_textures() -> void
  {
    _image_plane_renderer.initialize();

    const auto images = std::array<sara::Image<sara::Rgb8>, 2>{
        sara::imread<sara::Rgb8>(
            (_program_dir_path / "data/image-omni.png").string()),
        sara::imread<sara::Rgb8>(
            (_program_dir_path / "data/image-pinhole.png").string()),
    };

    auto& image_textures = _image_plane_renderer._textures;
    image_textures.resize(2);
    for (auto i = 0; i < 2; ++i)
    {
      // Transfer the CPU image data to the GPU texture.
      const auto texture_unit = i;
      image_textures[i].set_texture(images[i], texture_unit);

      // Geometry
      image_textures[i]._model_view.setIdentity();
      image_textures[i]._projection.setIdentity();
      // Dummy transformation just to make things to see.
      image_textures[i]._model_view(0, 3) += i * 0.5f;
      image_textures[i]._model_view(2, 3) -= i * 0.5f;
    }
  }

  auto initialize_lines() -> void
  {
    auto& gl_lines = _line_renderer._lines;
    gl_lines.resize(2);

    const auto image_sizes = Eigen::Vector2f{1920.f, 1080.f};

    auto line_data = LineRenderer::LineHostData{};

    // Draw a quad.
    line_data.add_line_segment_in_pixel_coordinates(  //
        {0.f, 0.f}, {1920.f, 0.f}, image_sizes, 3.f, 1.f);
    line_data.add_line_segment_in_pixel_coordinates(
        {1920.f, 0.f}, {1920.f, 1080.f}, image_sizes, 3.f, 1.f);
    line_data.add_line_segment_in_pixel_coordinates(
        {1920.f, 1080.f}, {0.f, 1080.f}, image_sizes, 3.f, 1.f);
    line_data.add_line_segment_in_pixel_coordinates(  //
        {0.f, 1080.f}, {0.f, 0.f}, image_sizes, 3.f, 1.f);
    gl_lines[0].set_data(line_data);
    gl_lines[0]._color << .8f, .0f, .0f, .5f;

    line_data.clear();

    // Draw a triangle.
    line_data.add_line_segment_in_pixel_coordinates(  //
        {0.f, 0.f}, {1920.f, 0.f}, image_sizes, 3.f, 1.f);
    line_data.add_line_segment_in_pixel_coordinates(
        {1920.f, 0.f}, {1920.f, 1080.f}, image_sizes, 3.f, 1.f);
    line_data.add_line_segment_in_pixel_coordinates(  //
        {1920.f, 1080.f}, {0.f, 0.f}, image_sizes, 3.f, 1.f);
    gl_lines[1].set_data(line_data);
    gl_lines[1]._color << .0f, .8f, .0f, .5f;
  }

  auto cleanup_gl_objects() -> void
  {
    // Destroy the shaders and quad geometry data.
    _image_plane_renderer.destroy_gl_objects();
    _line_renderer.destroy_gl_objects();

    // Destroy the image textures.
    auto& image_textures = _image_plane_renderer._textures;
    for (auto i = 0u; i < image_textures.size(); ++i)
      image_textures[i].destroy();
    image_textures.clear();

    // Clear the list of image textures.
    image_textures.clear();

    // Destroy the lines.
    auto& gl_lines = _line_renderer._lines;
    for (auto i = 0u; i < gl_lines.size(); ++i)
      gl_lines[i].destroy();
    gl_lines.clear();
  }

  auto terminate() -> void
  {
    cleanup_gl_objects();
    glfwTerminate();
  }

private:
  GLFWwindow* _window = nullptr;
  Eigen::Vector2i _window_sizes = Eigen::Vector2i::Zero();
#if defined(_WIN32)
  int _high_dpi_scale_factor;
#endif

  fs::path _program_dir_path = ".";

  ImagePlaneRenderer _image_plane_renderer;
  LineRenderer _line_renderer;

  static std::unique_ptr<GLFWApp> _instance;
};

std::unique_ptr<GLFWApp> GLFWApp::_instance = nullptr;

#if defined(__EMSCRIPTEN__)
auto render_frame_for_emscripten() -> void
{
  GLFWApp::instance().render_frame();
}
#endif


int main(int, [[maybe_unused]] char** argv)
{
  try
  {
    auto& app = GLFWApp::instantiate({800, 600}, "Line Renderer");
    app.initialize(fs::path{argv[0]}.parent_path());
    app.run();
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
