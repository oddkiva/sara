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
#include <DO/Kalpana/EasyGL/Objects/ColoredPointCloud.hpp>
#include <DO/Kalpana/EasyGL/Renderer/CheckerboardRenderer.hpp>
#include <DO/Kalpana/EasyGL/Renderer/ColoredPointCloudRenderer.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>

#include <GLFW/glfw3.h>

#include "GlfwUtilities.hpp"

#include <map>


using namespace DO::Sara;


namespace k = DO::Kalpana;
namespace kgl = k::GL;
namespace sara = DO::Sara;


auto resize_framebuffer(GLFWwindow*, int width, int height)
{
  // make sure the viewport matches the new window dimensions; note that width
  // and height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}

inline auto init_glfw_boilerplate()
{
  // Initialize the windows manager.
  if (!glfwInit())
    throw std::runtime_error{"Error: failed to initialize GLFW!"};

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
}

inline auto init_glew_boilerplate()
{
#if !defined(__APPLE__)
  // Initialize GLEW.
  auto err = glewInit();
  if (err != GLEW_OK)
  {
    std::cerr << sara::format("Error: failed to initialize GLEW: %s",
                              glewGetErrorString(err))
              << std::endl;
  }
#endif
}


auto read_point_cloud(const std::string& h5_filepath) -> Tensor_<float, 2>
{
  auto h5_file = H5File{h5_filepath, H5F_ACC_RDONLY};

  auto coords = MatrixXd{};
  h5_file.read_dataset("points", coords);
  coords.matrix() *= -1;
  auto coords_tensorview =
      TensorView_<double, 2>{coords.data(), {coords.cols(), coords.rows()}};

  auto colors = Tensor_<double, 2>{};
  h5_file.read_dataset("colors", colors);

  // Concatenate the data.
  auto vertex_data = Tensor_<double, 2>{{coords.cols(), 6}};
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


int main()
{
  // ==========================================================================
  // Boilerplate code for display initialization.
  //
  init_glfw_boilerplate();

  // Create a window.
  static constexpr auto width = 800;
  static constexpr auto height = 600;
  const auto window = glfwCreateWindow(width, height,   //
                                 "Hello Camera",  //
                                 nullptr, nullptr);
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, resize_framebuffer);
  glfwSetKeyCallback(window, move_camera_from_keyboard);
  glfwSetCursorPosCallback(window, move_trackball);
  glfwSetMouseButtonCallback(window, use_trackball);

  init_glew_boilerplate();


  // ==========================================================================
  // Initialize data on OpenGL side.
  auto point_cloud = kgl::ColoredPointCloud{};
  point_cloud.initialize();
  point_cloud.upload_host_data_to_gl(make_point_cloud());

  // Object renderers on OpenGL side.
  auto point_cloud_renderer = kgl::ColoredPointCloudRenderer{};
  point_cloud_renderer.initialize();
  auto checkerboard_renderer = kgl::CheckerboardRenderer(100, 100);

  // ==========================================================================
  // Setup options for point cloud rendering.
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_PROGRAM_POINT_SIZE);
  // Default background color.
  glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
  // You absolutely need this for 3D objects!
  glEnable(GL_DEPTH_TEST);
  glfwSwapInterval(1);

  // ==========================================================================
  // Model-view-projection matrix stack.
  //
  // Initialize the projection matrix once for all.
  const Matrix4f projection = k::perspective(45.f, 800.f / 600.f, .1f, 1000.f);
  // Transform matrix.
  const Transform<float, 3, Eigen::Projective> transform =
      Transform<float, 3, Eigen::Projective>::Identity();

  // Display image.
  while (!glfwWindowShouldClose(window))
  {
    // Calculate the elapsed time.
    gtime.update();

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      glfwSetWindowShouldClose(window, true);

    // Camera interaction with the trackball.
    // auto view_matrix = camera.view_matrix();
    Eigen::Matrix3f view_matrix_33 =
        trackball.rotation().toRotationMatrix().cast<float>();
    Eigen::Matrix4f view_matrix = Eigen::Matrix4f::Identity();
    view_matrix.topLeftCorner(3, 3) = view_matrix_33;
    view_matrix.col(3).head(3) = camera.position;

    Transform<float, 3, Eigen::Projective> scale_point_cloud =
        Transform<float, 3, Eigen::Projective>::Identity();
    scale_point_cloud.scale(scale);

    // Important.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw the checkerboard.
    if (show_checkerboard)
      checkerboard_renderer.render(transform.matrix(),    //
                                   view_matrix.matrix(),  //
                                   projection.matrix());

    // Draw point cloud.
    static constexpr auto point_size = 5.f;
    point_cloud_renderer.render(point_cloud, point_size,  //
                                transform.matrix(),       //
                                view_matrix.matrix(),     //
                                projection.matrix());

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // Destroy object renderers.
  checkerboard_renderer.destroy();
  point_cloud_renderer.destroy();

  // Destroy geometry data.
  point_cloud.destroy();

  // Clean up resources.
  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
