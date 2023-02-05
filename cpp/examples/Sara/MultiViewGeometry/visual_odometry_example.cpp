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
#include <DO/Kalpana/EasyGL/Objects/ColoredPointCloud.hpp>
#include <DO/Kalpana/EasyGL/Objects/TexturedImage.hpp>
#include <DO/Kalpana/EasyGL/Objects/TexturedQuad.hpp>
#include <DO/Kalpana/EasyGL/Renderer/ColoredPointCloudRenderer.hpp>
#include <DO/Kalpana/EasyGL/Renderer/TextureRenderer.hpp>
#include <DO/Kalpana/Math/Projection.hpp>
#include <DO/Kalpana/Math/Viewport.hpp>

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
#include <string_view>

#if defined(_OPENMP)
#  include <omp.h>
#endif


namespace fs = std::filesystem;
namespace sara = DO::Sara;
namespace k = DO::Kalpana;
namespace kgl = DO::Kalpana::GL;


class VideoStreamBlock
{
public:
  VideoStreamBlock() = default;

  VideoStreamBlock(const fs::path& video_path)
  {
    open(video_path);
  }

  auto open(const fs::path& video_path) -> void
  {
    _frame_index = -1;
    _video_stream.close();
    _video_stream.open(video_path);
    _rgb8.swap(_video_stream.frame());
    _gray32f.resize(_video_stream.sizes());
  }

  auto read() -> bool
  {
    const auto read_new_frame = _video_stream.read();
    sara::from_rgb8_to_gray32f(_rgb8, _gray32f);
    if (read_new_frame)
      ++_frame_index;
    return read_new_frame;
  }

  auto frame_rgb8() const -> const sara::ImageView<sara::Rgb8>&
  {
    return _rgb8;
  }

  auto frame_gray32f() const -> const sara::ImageView<float>&
  {
    return _gray32f;
  }

  auto sizes() const -> Eigen::Vector2i
  {
    return _video_stream.sizes();
  }

  auto width() const -> int
  {
    return _video_stream.width();
  }

  auto height() const -> int
  {
    return _video_stream.height();
  }

  auto skip() const -> bool
  {
    return _frame_index % (_frame_skip + 1) != 0;
  }

private:
  sara::VideoStream _video_stream;
  sara::ImageView<sara::Rgb8> _rgb8;
  sara::Image<float> _gray32f;

  int _frame_skip = 2;
  int _frame_index = -1;
};


class ImageCorrectionBlock
{
public:
  ImageCorrectionBlock() = default;

  ImageCorrectionBlock(
      const sara::v2::BrownConradyDistortionModel<double>& camera,
      const Eigen::Vector2i& image_sizes)
  {
    std::tie(_umap, _vmap) = sara::undistortion_map(camera, image_sizes);
    _rgb8_undistorted.resize(image_sizes);
    _gray32f_undistorted.resize(image_sizes);
  }

  auto undistort(const sara::ImageView<float>& gray32f) -> void
  {
    warp(_umap, _vmap, gray32f, _gray32f_undistorted);
  }

  auto undistort(const sara::ImageView<sara::Rgb8>& rgb8) -> void
  {
    warp(_umap, _vmap, rgb8, _rgb8_undistorted);
  }

  auto frame_gray32f() const -> const sara::ImageView<float>&
  {
    return _gray32f_undistorted;
  }

  auto frame_rgb8() const -> const sara::ImageView<sara::Rgb8>&
  {
    return _rgb8_undistorted;
  }

private:
  sara::Image<float> _umap;
  sara::Image<float> _vmap;
  sara::Image<sara::Rgb8> _rgb8_undistorted;
  sara::Image<float> _gray32f_undistorted;
};


struct FeatureTrackingBlock
{
  float sift_nn_ratio = 0.6f;
  sara::ImagePyramidParams image_pyr_params = sara::ImagePyramidParams(0);

  std::array<sara::Image<sara::Rgb8>, 2> _rgb8_images;
  std::array<sara::KeypointList<sara::OERegion, float>, 2> keys;
  std::vector<sara::Match> matches;

  auto detect_features(const sara::ImageView<float>& frame) -> void
  {
    sara::print_stage("Computing keypoints...");
    std::swap(keys[0], keys[1]);
    keys[1] = sara::compute_sift_keypoints(frame, image_pyr_params);
  }

  auto match_features() -> void
  {
    sara::print_stage("Matching keypoints...");
    if (sara::features(keys[0]).empty() || sara::features(keys[1]).empty())
      return;

    matches = sara::match(keys[0], keys[1], sift_nn_ratio);
    // Put a hard limit of 1000 matches to scale.
    if (matches.size() > 1000)
      matches.resize(1000);
  }
};

struct RelativePoseBlock
{
  int ransac_iterations_max = 500;
  double ransac_confidence = 0.999;
  double err_thres = 2.;

  const sara::RelativePoseSolver<sara::NisterFivePointAlgorithm> _solver;
  sara::CheiralAndEpipolarConsistency _inlier_predicate;

  sara::PointCorrespondenceList<double> _X;

  Eigen::Matrix3d _K;
  Eigen::Matrix3d _K_inv;
  sara::Tensor_<bool, 1> _inliers;
  sara::Tensor_<int, 1> _sample_best;
  sara::TwoViewGeometry _geometry;

  auto configure(const sara::v2::BrownConradyDistortionModel<double>& camera)
      -> void
  {
    _K = camera.calibration_matrix();
    _K_inv = _K.inverse();

    _inlier_predicate.distance.K1_inv = _K_inv;
    _inlier_predicate.distance.K2_inv = _K_inv;
    _inlier_predicate.err_threshold = err_thres;
  }

  auto estimate_relative_pose(
      const std::array<sara::KeypointList<sara::OERegion, float>, 2> keys,
      const std::vector<sara::Match>& matches) -> void
  {
    sara::print_stage("Estimating the relative pose...");
    if (matches.empty())
      return;

    const auto& f0 = sara::features(keys[0]);
    const auto& f1 = sara::features(keys[1]);
    const auto u =
        std::array{sara::homogeneous(sara::extract_centers(f0)).cast<double>(),
                   sara::homogeneous(sara::extract_centers(f1)).cast<double>()};
    // List the matches as a 2D-tensor where each row encodes a match 'm' as a
    // pair of point indices (i, j).
    const auto M = sara::to_tensor(matches);

    _X = sara::PointCorrespondenceList{M, u[0], u[1]};
    auto data_normalizer =
        std::make_optional(sara::Normalizer<sara::TwoViewGeometry>{_K, _K});

    std::tie(_geometry, _inliers, _sample_best) =
        sara::v2::ransac(_X, _solver, _inlier_predicate, ransac_iterations_max,
                         ransac_confidence, data_normalizer, true);
    SARA_DEBUG << "Geometry =\n" << _geometry << std::endl;
    SARA_DEBUG << "inliers count = " << _inliers.flat_array().count()
               << std::endl;
  }
};

struct TriangulationBlock
{
  sara::TwoViewGeometry geometry;
  sara::Tensor_<double, 2> colors;
  sara::Tensor_<double, 2> _colored_point_cloud;

  auto triangulate(const sara::PinholeCameraDecomposition& C1,
                   const sara::PinholeCameraDecomposition& C2,
                   const Eigen::Matrix3d& K, const Eigen::Matrix3d& K_inv,
                   const sara::PointCorrespondenceList<double>& X,
                   const sara::TensorView_<bool, 1>& inliers) -> void
  {
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
        const auto [Xj, s1j, s2j] = sara::triangulate_single_point_linear_eigen(
            C1.matrix(), C2.matrix(), u1, u2);
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
#if 0
    colors = sara::extract_colors(frames_rgb_undist[0],
                                  frames_rgb_undist[1],  //
                                  geometry);
#endif
  }
};


struct Pipeline
{
  Pipeline() = default;

  auto set_config(const fs::path& video_path,
                  const sara::v2::BrownConradyDistortionModel<double> camera)
      -> void
  {
    _video_streamer.open(video_path);
    _camera = camera;

    _image_corrector = ImageCorrectionBlock{_camera, _video_streamer.sizes()};

    _relative_pose_estimator.configure(_camera);
  }

  auto read() -> bool
  {
    return _video_streamer.read();
  }

  auto process() -> void
  {
    if (_video_streamer.skip())
      return;

    _image_corrector.undistort(_video_streamer.frame_rgb8());
    _image_corrector.undistort(_video_streamer.frame_gray32f());

    // N.B.: detect the features on the **undistorted** image.
    _feature_tracker.detect_features(_image_corrector.frame_gray32f());
    _feature_tracker.match_features();

    _relative_pose_estimator.estimate_relative_pose(_feature_tracker.keys,
                                                    _feature_tracker.matches);

    _triangulator.triangulate(
        _relative_pose_estimator._geometry.C1,
        _relative_pose_estimator._geometry.C2,  //
        _relative_pose_estimator._K, _relative_pose_estimator._K_inv,
        _relative_pose_estimator._X, _relative_pose_estimator._inliers);
  }

  auto make_display_frame() const -> sara::Image<sara::Rgb8>
  {
    sara::Image<sara::Rgb8> display = _image_corrector.frame_rgb8();
    const auto& matches = _feature_tracker.matches;
    const auto& inliers = _relative_pose_estimator._inliers;
#pragma omp parallel for
    for (auto m = 0u; m < matches.size(); ++m)
    {
      if (!inliers(m))
        continue;
      const auto& match = matches[m];
      sara::draw(display, match.x(), sara::Blue8);
      sara::draw(display, match.y(), sara::Cyan8);
      sara::draw_arrow(display, match.x_pos(), match.y_pos(), sara::Yellow8);
    }

    return display;
  }

  VideoStreamBlock _video_streamer;
  sara::v2::BrownConradyDistortionModel<double> _camera;

  ImageCorrectionBlock _image_corrector;
  FeatureTrackingBlock _feature_tracker;
  RelativePoseBlock _relative_pose_estimator;
  TriangulationBlock _triangulator;
};


class SingleWindowApp
{
public:
  SingleWindowApp(const Eigen::Vector2i& sizes, const std::string& title)
    : _window_sizes{sizes}
  {
    // Init GLFW.
    init_glfw();

    // Create a window.
    _window = glfwCreateWindow(sizes.x(), sizes.y(),  //
                               title.c_str(),         //
                               nullptr, nullptr);

    // Initialize the point cloud viewport.
    _point_cloud_viewport.top_left.setZero();
    _point_cloud_viewport.sizes << sizes.x() / 2, sizes.y();

    // Initialize the video viewport.
    _video_viewport.top_left << sizes.x() / 2, 0;
    _video_viewport.sizes << sizes.x() / 2, sizes.y();

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

    // Background color.
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    // Display image.
    glfwSwapInterval(1);
    while (!glfwWindowShouldClose(_window))
    {
      if (!_pipeline.read())
        break;

      _pipeline.process();

      // Clear the color buffer and the buffer testing.
      glClear(GL_COLOR_BUFFER_BIT);

      render_video();

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

  auto upload_point_cloud_data_to_opengl(
      const sara::TensorView_<float, 2>& point_cloud) -> void
  {
    _point_cloud.upload_host_data_to_gl(
        _pipeline._triangulator._colored_point_cloud);
  }

  auto render_video() -> void
  {
    // Render on the right half of the window surface.
    glViewport(_video_viewport.top_left.x(), _video_viewport.top_left.y(),  //
               _video_viewport.sizes.x(), _video_viewport.sizes.y());
    // Transfer the CPU image frame data to the OpenGL texture.
    // _texture.reset(_pipeline._video_stream.frame_rgb8());
    _texture.reset(_pipeline.make_display_frame());
    // Render the texture on the quad.
    auto model_view = Eigen::Transform<float, 3, Eigen::Projective>{};
    model_view.setIdentity();
    _texture_renderer.render(_texture, _quad, model_view.matrix(), _projection);
  }

  auto render_point_cloud() -> void
  {
    glViewport(
        _point_cloud_viewport.top_left.x(), _point_cloud_viewport.top_left.y(),
        _point_cloud_viewport.sizes.x(), _point_cloud_viewport.sizes.y());

    // CAVEAT: re-express the point cloud in OpenGL axis convention.
    auto from_cam_to_gl = Eigen::Transform<float, 3, Eigen::Projective>{};
    from_cam_to_gl.setIdentity();
    // clang-format off
    from_cam_to_gl.matrix().topLeftCorner<3, 3>() <<
      1,  0,  0,
      0, -1,  0,
      0,  0, -1;
    // clang-format on

    // Recalculate the projection matrix for the point cloud.
    const auto aspect_ratio = _point_cloud_viewport.aspect_ratio();
    const Eigen::Matrix4f projection = k::perspective(60.f,          //
                                                      aspect_ratio,  //
                                                      .5f,           //
                                                      200.f);

    // Update the model view matrix.
    const Eigen::Matrix4f model_view = Eigen::Matrix4f::Identity();

    // Render the point cloud.
    _point_cloud_renderer.render(_point_cloud, _point_size,
                                 from_cam_to_gl.matrix(),  //
                                 model_view, projection);
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
    auto& self = get_self(window);
    self._window_sizes << width, height;

    // Reset the viewport sizes
    self._video_viewport.top_left << width / 2, 0;
    self._video_viewport.sizes << width / 2, height;
    // Update the current projection matrix.
    auto scale = 0.5f;
    if (self._video_viewport.width() < self._pipeline._video_streamer.width())
      scale *= static_cast<float>(self._pipeline._video_streamer.width()) /
               self._video_viewport.width();
    self._projection = self._video_viewport.orthographic_projection(scale);
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
    {
      const auto err_str =
          reinterpret_cast<const char*>(glewGetErrorString(err));
      throw std::runtime_error{fmt::format(
          "Error: failed to initialize GLEW: {}", std::string_view{err_str})};
    }
#endif
  }

private:
  GLFWwindow* _window = nullptr;
  Eigen::Vector2i _window_sizes = -Eigen::Vector2i::Ones();

  Pipeline _pipeline;

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
  // kgl::Camera _point_cloud_camera;
  float _point_size = 3.f;
};


auto main(int const argc, char** const argv) -> int
{
#if defined(_OPENMP)
  const auto num_threads = omp_get_max_threads();
  omp_set_num_threads(num_threads);
  Eigen::setNbThreads(num_threads);
#endif

  if (argc < 2)
  {
    std::cout << fmt::format("Usage: {} VIDEO_PATH\n",
                             std::string_view{argv[0]});
    return 1;
  }

  const auto video_path = fs::path{argv[1]};
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
      0.05952465745165465,
      -0.007947847982157091;
    // clang-format on
    camera.p() << -0.0003137658969742134, 0.00021943576376532096;
  }

  auto app = SingleWindowApp{{800, 600}, "Odometry: " + video_path.string()};
  app.set_config(video_path, camera);
  app.run();
  app.terminate();

  return 0;
}
