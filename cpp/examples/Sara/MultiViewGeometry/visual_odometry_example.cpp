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
#include <DO/Kalpana/EasyGL/Objects/TexturedImage.hpp>
#include <DO/Kalpana/EasyGL/Objects/TexturedQuad.hpp>
#include <DO/Kalpana/EasyGL/Renderer/TextureRenderer.hpp>
#include <DO/Kalpana/Math/Projection.hpp>

#include <DO/Sara/FeatureDetectors/SIFT.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Interpolation.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/v2/BrownConradyCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/v2/OmnidirectionalCamera.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/ErrorMeasures.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/InlierPredicates.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/RelativePoseSolver.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>
#include <DO/Sara/RANSAC/RANSACv2.hpp>
#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>
#include <DO/Sara/SfM/BuildingBlocks/Triangulation.hpp>
#include <DO/Sara/VideoIO.hpp>
#include <DO/Sara/Visualization/Features/Draw.hpp>

#include "ImageWarpUtilities.hpp"

#ifdef _WIN32
#  include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <fmt/format.h>

#include <filesystem>

#if defined(_OPENMP)
#  include <omp.h>
#endif


namespace fs = std::filesystem;
namespace sara = DO::Sara;
namespace k = DO::Kalpana;
namespace kgl = DO::Kalpana::GL;


class SingleWindowApp
{
public:
  SingleWindowApp(const Eigen::Vector2i& sizes, const std::string& title)
    : _window_sizes{sizes}
  {
    // Init GLFW.
    init_glfw();

    // Create a GLFW window.
    _window = create_glfw_window(sizes, title);

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

  auto run() -> void
  {
    // Projection-model-view matrices.
    auto model_view = Eigen::Transform<float, 3, Eigen::Projective>{};
    model_view.setIdentity();

    const auto win_aspect_ratio =
        static_cast<float>(_window_sizes.x()) / _window_sizes.y();
    _projection = k::orthographic(            //
        -win_aspect_ratio, win_aspect_ratio,  //
        -1.f, 1.f,                            //
        -1.f, 1.f);

    // Video state.
    auto frame_index = -1;

    // Render on the whole window surface.
    glViewport(0, 0, _window_sizes.x(), _window_sizes.y());
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
private:
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

  static auto window_size_callback(GLFWwindow* window, const int width,
                                   const int height) -> void
  {
    auto& app = get_self(window);
    app._window_sizes << width, height;
    const auto aspect_ratio = static_cast<float>(width) / height;
    app._projection = k::orthographic(-0.5f * aspect_ratio, 0.5f * aspect_ratio,
                                      -0.5f, 0.5f, -0.5f, 0.5f);
  }

private:
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

private:
  GLFWwindow* _window = nullptr;
  Eigen::Vector2i _window_sizes = -Eigen::Vector2i::Ones();

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
    std::cout << fmt::format("Usage: {} VIDEO_PATH\n",
                             std::string_view{argv[0]});
    return 1;
  }

  const auto video_path = fs::path{argv[1]};
  auto app = SingleWindowApp{{800, 600}, "Odometry: " + video_path.string()};

  app.terminate();

  return 0;
}


#if 0
auto main(int const argc, char** const argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}

auto sara_graphics_main(int, char**) -> int
{
#  if defined(_OPENMP)
  const auto num_threads = omp_get_max_threads();
  omp_set_num_threads(num_threads);
  Eigen::setNbThreads(num_threads);
#  endif

  using namespace std::string_literals;
#  if defined(__APPLE__)
  const auto video_path = "/Users/david/Desktop/Datasets/sample-1.mp4"s;
#  else
  const auto video_path = "/home/david/Desktop/Datasets/sample-1.mp4"s;
#  endif

  auto camera = sara::v2::BrownConradyDistortionModel<double>{};
  camera.fx() = 917.2878392016245;
  camera.fy() = 917.2878392016245;
  camera.shear() = 0.;
  camera.u0() = 960.;
  camera.v0() = 540.;
  camera.k() << -0.2338367557617234, 0.05952465745165465, -0.007947847982157091;
  camera.p() << -0.0003137658969742134, 0.00021943576376532096;

  const auto K = camera.calibration_matrix();
  const Eigen::Matrix3d K_inv = K.inverse();
  SARA_DEBUG << "K =\n" << K << std::endl;
  SARA_DEBUG << "K^{-1} =\n" << K_inv << std::endl;

  auto video_stream = sara::VideoStream{video_path};

  auto frame_rgb8 = video_stream.frame();
  auto frame_gray32f = sara::Image<float>{video_stream.sizes()};

  auto frames_rgb_undist = std::array<sara::Image<sara::Rgb8>, 2>{};
  auto frames_work = std::array<sara::Image<float>, 2>{};
  auto& frame_undistorted = frames_work[1];

  const auto coords_map = sara::undistortion_map(camera, video_stream.sizes());
  const auto& [umap, vmap] = coords_map;

  constexpr auto sift_nn_ratio = 0.6f;
  constexpr auto ransac_iterations_max = 200;
  constexpr auto ransac_confidence = 0.999;
  constexpr auto err_thres = 2.;
  const auto image_pyr_params = sara::ImagePyramidParams(0);
  auto solver = sara::RelativePoseSolver<sara::NisterFivePointAlgorithm>{};
  auto inlier_predicate = sara::CheiralAndEpipolarConsistency{};
  {
    inlier_predicate.distance.K1_inv = K_inv;
    inlier_predicate.distance.K2_inv = K_inv;
    inlier_predicate.err_threshold = err_thres;
  }

  auto keys = std::array<sara::KeypointList<sara::OERegion, float>, 2>{};
  auto E = sara::EssentialMatrix{};
  auto F = sara::FundamentalMatrix{};
  auto matches = std::vector<sara::Match>{};
  auto inliers = sara::Tensor_<bool, 1>{};
  auto sample_best = sara::Tensor_<int, 1>{};
  auto geometry = sara::TwoViewGeometry{};


  sara::create_window(video_stream.sizes());
  sara::set_antialiasing();

  auto frame_index = -1;
  while (video_stream.read())
  {
    ++frame_index;
    if (frame_index % 5 != 0)
      continue;
    SARA_CHECK(frame_index);

    frames_rgb_undist[0].swap(frames_rgb_undist[1]);
    frames_work[0].swap(frames_work[1]);

    if (frame_undistorted.sizes() != frame_rgb8.sizes())
      frame_undistorted.resize(frame_rgb8.sizes());
    if (frames_rgb_undist[1].sizes() != frame_rgb8.sizes())
      frames_rgb_undist[1].resize(frame_rgb8.sizes());
    sara::from_rgb8_to_gray32f(frame_rgb8, frame_gray32f);
    warp(umap, vmap, frame_rgb8, frames_rgb_undist[1]);
    warp(umap, vmap, frame_gray32f, frame_undistorted);
    auto display = frames_rgb_undist[1];

    sara::print_stage("Computing keypoints...");
    std::swap(keys[0], keys[1]);
    keys[1] = sara::compute_sift_keypoints(frame_undistorted, image_pyr_params);

    const auto& f0 = sara::features(keys[0]);
    const auto& f1 = sara::features(keys[1]);
    const auto do_relative_pose_estimation = !f0.empty() && !f1.empty();

    if (do_relative_pose_estimation)
    {
      sara::print_stage("Matching keypoints...");
      matches = match(keys[0], keys[1], sift_nn_ratio);
      // Put a hard limit of 1000 matches to scale.
      if (matches.size() > 1000)
        matches.resize(1000);

      sara::print_stage("Estimating the relative pose...");
      const auto u = std::array{
          sara::homogeneous(sara::extract_centers(f0)).cast<double>(),
          sara::homogeneous(sara::extract_centers(f1)).cast<double>()};
      // List the matches as a 2D-tensor where each row encodes a match 'm' as a
      // pair of point indices (i, j).
      const auto M = sara::to_tensor(matches);
      const auto X = sara::PointCorrespondenceList{M, u[0], u[1]};
      auto data_normalizer =
          std::make_optional(sara::Normalizer<sara::TwoViewGeometry>{K, K});
      auto sample_best = sara::Tensor_<int, 1>{};
      std::tie(geometry, inliers, sample_best) =
          sara::v2::ransac(X, solver, inlier_predicate, ransac_iterations_max,
                           ransac_confidence, data_normalizer, true);
      SARA_DEBUG << "Geometry =\n" << geometry << std::endl;
      SARA_DEBUG << "inliers count = " << inliers.flat_array().count()
                 << std::endl;

      // Retrieve all the 3D points by triangulation.
      sara::print_stage("Retriangulating the inliers...");
      auto& points = geometry.X;
      auto& s1 = geometry.scales1;
      auto& s2 = geometry.scales2;
      points.resize(4, inliers.flat_array().count());
      s1.resize(inliers.flat_array().count());
      s2.resize(inliers.flat_array().count());
      auto cheiral_inlier_count = 0;
      {
        auto& j = cheiral_inlier_count;
        for (auto i = 0; i < inliers.size(0); ++i)
        {
          if (!inliers(i))
            continue;

          const Eigen::Vector3d u1 = K_inv * X[i][0].vector();
          const Eigen::Vector3d u2 = K_inv * X[i][1].vector();
          const auto [Xj, s1j, s2j] =
              sara::triangulate_single_point_linear_eigen(
                  geometry.C1.matrix(), geometry.C2.matrix(), u1, u2);
          const auto cheiral = s1j > 0 && s2j > 0;
          if (!cheiral)
            continue;

          // Also we want z in [0, 200] meters max...
          // We want to avoid 3D point corresponding to the sky...
          Eigen::Vector4d Xjh = Xj.hnormalized().homogeneous();
          const auto reasonable = 0 < Xjh.z() && Xjh.z() < 200;
          if (!reasonable)
            continue;

          points.col(j) = Xjh;
          s1(j) = s1j;
          s2(j) = s2j;
          ++j;
        }
        SARA_CHECK(cheiral_inlier_count);
        points = points.leftCols(cheiral_inlier_count);
        s1 = s1.head(cheiral_inlier_count);
        s2 = s2.head(cheiral_inlier_count);
      }

      sara::print_stage("Extracting the colors...");
      geometry.C1.K = K;
      geometry.C2.K = K;
      auto colors = sara::extract_colors(frames_rgb_undist[0],
                                         frames_rgb_undist[1],  //
                                         geometry);

      sara::print_stage("Saving to HDF5");
      {
#  if defined(__APPLE__)
        const auto geometry_h5_filepath = "/Users/david/Desktop/geometry" +
#  else
        const auto geometry_h5_filepath = "/home/david/Desktop/geometry" +
#  endif
                                          std::to_string(frame_index) + ".h5"s;
        auto geometry_h5_file =
            sara::H5File{geometry_h5_filepath, H5F_ACC_TRUNC};
        sara::save_to_hdf5(geometry_h5_file, geometry, colors);
      }

      F.matrix() = K_inv.transpose() * E.matrix() * K_inv;

      sara::print_stage("Draw...");
      for (auto m = 0u; m < matches.size(); ++m)
      {
        if (!inliers(m))
          continue;
        const auto& match = matches[m];
        sara::draw(display, match.x(), sara::Blue8);
        sara::draw(display, match.y(), sara::Cyan8);
        sara::draw_arrow(display, match.x_pos(), match.y_pos(), sara::Yellow8);
      }

      sara::display(display);
      // if (sara::get_key() == sara::KEY_ESCAPE)
      //   break;
    }
  }

  return 0;
}
#endif
