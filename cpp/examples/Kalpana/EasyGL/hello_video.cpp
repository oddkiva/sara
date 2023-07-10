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
#include <DO/Kalpana/EasyGL/Objects/TexturedImage.hpp>
#include <DO/Kalpana/EasyGL/Objects/TexturedQuad.hpp>
#include <DO/Kalpana/EasyGL/Renderer/TextureRenderer.hpp>
#include <DO/Kalpana/Math/Projection.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/VideoIO.hpp>

#ifdef _WIN32
#  include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <filesystem>


namespace k = DO::Kalpana;
namespace kgl = DO::Kalpana::GL;
namespace sara = DO::Sara;
namespace fs = std::filesystem;


struct SingleWindowApp
{
public:
  SingleWindowApp(const Eigen::Vector2i& sizes, const std::string& title)
  {
    // Init GLFW.
    init_glfw();

    // Create a GLFW window.
    _window = create_glfw_window(sizes, title);
    _fb_sizes = get_framebuffer_sizes();

    // Prepare OpenGL first before any OpenGL calls.
    init_opengl();

    // The magic function.
    glfwSetWindowUserPointer(_window, this);
    // Register callbacks.
    glfwSetWindowSizeCallback(_window, window_size_callback);
  }

  //! @brief Note: RAII does not work on OpenGL applications.
  //!
  //! So the destructor gets a default implementation and we neeed to explicitly
  //! call the terminate method.
  ~SingleWindowApp() = default;

  auto open_video(const fs::path& video_path) -> void
  {
    _video_stream.open(video_path.string());
  }

  auto init_opengl() -> void
  {
    // GLFW context...
    glfwMakeContextCurrent(_window);

    // Init OpenGL extensions.
    init_glew();
  }

  auto init_gl_resources() -> void
  {
    _texture.initialize(_video_stream.frame(), 0);

    const auto& w = _video_stream.width();
    const auto& h = _video_stream.height();
    const auto aspect_ratio = static_cast<float>(w) / h;
    auto vertices = _quad.host_vertices().matrix();
    vertices.col(0) *= aspect_ratio;
    _quad.initialize();

    _texture_renderer.initialize();
  }

  auto deinit_gl_resources() -> void
  {
    _texture.destroy();
    _quad.destroy();
    _texture_renderer.destroy();
  }

  auto run() -> void
  {
    // Projection-model-view matrices.
    auto model_view = Eigen::Transform<float, 3, Eigen::Projective>{};
    model_view.setIdentity();

    const auto fb_aspect_ratio =
        static_cast<float>(_fb_sizes.x()) / _fb_sizes.y();
    _projection = k::orthographic(                          //
        -0.5f * fb_aspect_ratio, 0.5f * fb_aspect_ratio,  //
        -0.5f, 0.5f,                                        //
        -0.5f, 0.5f);

    // Video state.
    auto frame_index = -1;

    // Background color.
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    // Display image.
    glfwSwapInterval(1);
    while (!glfwWindowShouldClose(_window))
    {
      if (!_video_stream.read())
        break;
      ++frame_index;

      // Clear the color buffer and the buffer testing.
      glClear(GL_COLOR_BUFFER_BIT);

      // Render on the whole window surface.
      glViewport(0, 0, _fb_sizes.x(), _fb_sizes.y());

      // Transfer the CPU image frame data to the OpenGL texture.
      _texture.reset(_video_stream.frame());
      // Render the texture on the quad.
      _texture_renderer.render(_texture, _quad, model_view.matrix(),
                               _projection);

      glfwSwapBuffers(_window);
      glfwPollEvents();
    }
  }

  auto terminate() -> void
  {
    // Destroy GL objects.
    deinit_gl_resources();

    // Destroy GLFW.
    if (_window != nullptr)
      glfwDestroyWindow(_window);
    glfwTerminate();
  }

private: /* callback functions */
  static auto window_size_callback(GLFWwindow* window, const int, const int)
      -> void
  {
    auto& app = get_self(window);
    auto& sizes = app._fb_sizes;
    sizes = app.get_framebuffer_sizes();
    const auto aspect_ratio = static_cast<float>(sizes.x()) / sizes.y();
    app._projection = k::orthographic(-0.5f * aspect_ratio, 0.5f * aspect_ratio,
                                      -0.5f, 0.5f,  //
                                      -0.5f, 0.5f);
  }

private: /* convenience free functions*/
  static auto init_glfw() -> void
  {
    // Initialize the windows manager.
    if (!glfwInit())
      throw std::runtime_error{"Error: failed to initialize GLFW!"};

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
  }

  static auto init_glew() -> void
  {
#ifndef __APPLE__
    // Initialize GLEW.
    const auto err = glewInit();
    if (err != GLEW_OK)
      throw std::runtime_error{sara::format(
          "Error: failed to initialize GLEW: %s", glewGetErrorString(err))};
#endif
  }

  static auto create_glfw_window(const Eigen::Vector2i& sizes,
                                 const std::string& title) -> GLFWwindow*
  {
    auto window = glfwCreateWindow(sizes.x(), sizes.y(),  //
                                   title.c_str(),         //
                                   nullptr, nullptr);
    return window;
  }

  static auto get_self(GLFWwindow* const window) -> SingleWindowApp&
  {
    const auto app_void_ptr = glfwGetWindowUserPointer(window);
    if (app_void_ptr == nullptr)
      throw std::runtime_error{
          "Please call glfwSetWindowUserPointer to register this window!"};
    const auto app_ptr = reinterpret_cast<SingleWindowApp*>(app_void_ptr);
    return *app_ptr;
  }

  auto get_framebuffer_sizes() const -> Eigen::Vector2i
  {
    auto sizes = Eigen::Vector2i{};
    glfwGetFramebufferSize(_window, &sizes.x(), &sizes.y());
    return sizes;
  }


private: /* data members */
  GLFWwindow* _window = nullptr;
  Eigen::Vector2i _fb_sizes = -Eigen::Vector2i::Ones();

  Eigen::Matrix4f _projection;

  // Our video stream.
  sara::VideoStream _video_stream;
  // What: our image texture.
  kgl::TexturedImage2D _texture;
  // Where: where to show our image texture.
  kgl::TexturedQuad _quad;
  // Texture renderer.
  kgl::TextureRenderer _texture_renderer;
};


auto main(int const argc, char** const argv) -> int
{
  if (argc < 2)
  {
    std::cout << "Usage: " << argv[0] << " VIDEO_FILEPATH" << std::endl;
    return EXIT_FAILURE;
  }

  auto app = SingleWindowApp({800, 600}, "Hello Video");

  // The order matters.
  app.open_video(fs::path{argv[1]});
  app.init_gl_resources();
  app.run();
  app.terminate();

  return EXIT_SUCCESS;
}
