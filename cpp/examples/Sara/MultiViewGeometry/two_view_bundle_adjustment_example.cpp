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

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/Logging/Logger.hpp>
#include <DO/Sara/MultiViewGeometry/BundleAdjustmentProblem.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/v2/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>
#include <DO/Sara/SfM/BuildingBlocks/PointCloudGenerator.hpp>
#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>
#include <DO/Sara/SfM/Graph/FeatureTracker.hpp>
#include <DO/Sara/SfM/Helpers.hpp>

#if defined(_WIN32)
#  pragma warning(push, 0)
#endif
#include <ceres/ceres.h>
#if defined(_WIN32)
#  pragma warning(pop)
#endif
#include <ceres/rotation.h>

#include <filesystem>


using namespace std;
using namespace std::string_literals;

namespace fs = std::filesystem;
namespace sara = DO::Sara;


#if 0
struct ReprojectionError
{
  ReprojectionError(double observed_x, double observed_y)
    : observed_x{observed_x}
    , observed_y{observed_y}
  {
  }

  template <typename T>
  bool operator()(const T* const camera,  // (1) camera parameters to optimize.
                  const T* const point,   // (2) 3D points to optimize
                  T* residuals) const
  {
    T p[3];

    auto camera_view = CameraModelView<T>{camera};

    // Rotate the point.
    ceres::AngleAxisRotatePoint(camera_view.angle_axis(), point, p);
    // Translate the point.
    p[0] += camera_view.t()[0];
    p[1] += camera_view.t()[1];
    p[2] += camera_view.t()[2];

    // Normalized camera coordinates.
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    // Apply second and fourth order order radial distortion.
    const auto& l1 = camera_view.l1();
    const auto& l2 = camera_view.l2();
    const auto r2 = xp * xp + yp * yp;
    const auto distortion = T(1) + r2 * (l1 + l2 * r2);
    xp *= distortion;
    yp *= distortion;

    // Apply the internal camera matrix.
    const auto predicted_x = camera_view.fx() * xp + camera_view.x0();
    const auto predicted_y = camera_view.fy() * yp + camera_view.y0();

    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  static ceres::CostFunction* create(const double observed_x,
                                     const double observed_y)
  {
    constexpr auto NumParams = 6 /* camera parameters */ + 3 /* points */;
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, NumParams, 3>{
        new ReprojectionError{observed_x, observed_y}};
  }

  double observed_x;
  double observed_y;
};
#endif


GRAPHICS_MAIN()
{
  auto& logger = sara::Logger::get();

  // Load the image data.
  SARA_LOGI(logger, "Loading images...");
  const auto data_dir_path = fs::path
  {
#if defined(__APPLE__)
    "/Users/oddkiva/Desktop/Datasets/sfm/castle_int"s
#else
    "/home/david/Desktop/Datasets/sfm/castle_int"s
#endif
  };
  const auto image_ids = std::array<std::string, 2>{"0000", "0001"};
  const auto image_paths = std::array{
      data_dir_path / (image_ids[0] + ".png"),
      data_dir_path / (image_ids[1] + ".png")  //
  };
  const auto images = std::array{
      sara::imread<sara::Rgb8>(image_paths[0]),  //
      sara::imread<sara::Rgb8>(image_paths[1])   //
  };

  // Load the calibration matrices.
  SARA_LOGI(logger, "Loading the internal camera matrices...");
  const auto K = std::array{
      sara::read_internal_camera_parameters(
          (data_dir_path / (image_ids[0] + ".png.K")).string())
          .cast<double>(),
      sara::read_internal_camera_parameters(
          (data_dir_path / (image_ids[1] + ".png.K")).string())
          .cast<double>()  //
  };
  for (auto i = 0; i < 2; ++i)
    SARA_LOGD(logger, "K[{}] =\n{}", i, K[i]);

  SARA_LOGI(logger, "Computing keypoints...");
  const auto image_pyr_params = sara::ImagePyramidParams(-1);
  auto abs_pose_data = std::array{
      sara::AbsolutePoseData{
          0,
          sara::compute_sift_keypoints(images[0].convert<float>(),
                                       image_pyr_params),
          sara::QuaternionBasedPose<double>::identity()  //
      },
      sara::AbsolutePoseData{
          1,
          sara::compute_sift_keypoints(images[1].convert<float>(),
                                       image_pyr_params),
          sara::QuaternionBasedPose<double>::nan()  //
      }                                             //
  };

  // Initialize the pose graph.
  SARA_LOGI(logger, "Initializing the vertices of the pose graph...");
  auto pose_graph = sara::CameraPoseGraph{};
  for (auto i = 0; i < 2; ++i)
    pose_graph.add_absolute_pose(std::move(abs_pose_data[i]));

  SARA_LOGI(logger, "Matching keypoints...");
  // The feature tracker.
  auto matches = sara::match(pose_graph[0].keypoints,  //
                             pose_graph[1].keypoints);

  SARA_LOGI(logger, "Performing data transformations...");
  // Invert the internal camera matrices.
  const auto K_inv = std::array<Eigen::Matrix3d, 2>{
      K[0].inverse(),
      K[1].inverse()  //
  };
  // Tensors of image coordinates.
  const auto& f0 = sara::features(pose_graph[0].keypoints);
  const auto& f1 = sara::features(pose_graph[1].keypoints);
  const auto u = std::array{
      sara::homogeneous(sara::extract_centers(f0)).cast<double>(),
      sara::homogeneous(sara::extract_centers(f1)).cast<double>()  //
  };
  // Tensors of camera coordinates.
  const auto un = std::array{
      sara::apply_transform(K_inv[0], u[0]),  //
      sara::apply_transform(K_inv[1], u[1])   //
  };
  static_assert(std::is_same_v<decltype(un[0]),  //
                               const sara::Tensor_<double, 2>&>);
  // List the matches as a 2D-tensor where each row encodes a match 'm' as a
  // pair of point indices (i, j).
  const auto M = sara::to_tensor(matches);
  const auto X = sara::PointCorrespondenceList{M, un[0], un[1]};

  SARA_LOGI(logger, "Estimating the essential matrix...");
  const auto num_samples = 1000;
  const auto err_thres = 1e-3;
  auto inlier_predicate =
      sara::InlierPredicate<sara::SampsonEpipolarDistance>{};
  inlier_predicate.err_threshold = err_thres;

  auto [E, inliers, sample_best] = sara::ransac(
      X, sara::NisterFivePointAlgorithm{}, inlier_predicate, num_samples);

  // Calculate the fundamental matrix.
  SARA_LOGI(logger, "Computing the fundamental matrix...");
  auto F = sara::FundamentalMatrix{};
  F.matrix() = K_inv[1].transpose() * E.matrix() * K_inv[0];

  // Extract the two-view geometry.
  SARA_LOGI(logger, "Estimating the two-view geometry...");
  auto two_view_geometry = sara::estimate_two_view_geometry(
      M, un[0], un[1], E, inliers, sample_best);
  two_view_geometry.C1.K = K[0];
  two_view_geometry.C2.K = K[1];

  auto rel_pose_data = sara::RelativePoseData{
      .matches = std::move(matches),
      .inliers = std::move(inliers),
      .motion =
          {
              .R = two_view_geometry.C2.R,  //
              .t = two_view_geometry.C2.t   //
          }                                 //
  };

  // Update the
  SARA_LOGI(logger,
            "Updating the absolute pose of vertex {} in the pose graph...", 1);
  pose_graph[1].pose = {
      .q = Eigen::Quaterniond{rel_pose_data.motion.R},
      .t = rel_pose_data.motion.t  //
  };
  SARA_LOGI(logger,
            "Adding the relative pose data for the edge ({}, {}) in the pose "
            "graph...",
            0, 1);
  const auto pose_edge =
      pose_graph.add_relative_pose(0, 1, std::move(rel_pose_data));

  SARA_LOGI(logger, "Populating the feature tracks...");
  auto feature_tracker = sara::FeatureTracker{};
  feature_tracker.update_feature_tracks(pose_graph, pose_edge);

  // 3. Recalculate the feature tracks that are still alive.
  const auto [tracks_alive, track_visibility_count] =
      feature_tracker.calculate_alive_feature_tracks(1);

  auto point_cloud = sara::PointCloudGenerator::PointCloud{};
  auto point_cloud_generator = sara::PointCloudGenerator{
      pose_graph, feature_tracker._feature_graph, point_cloud};

  auto pinhole_camera = sara::v2::PinholeCamera<double>{};
  point_cloud_generator.grow_point_cloud(tracks_alive, images[1], pose_edge,
                                         pinhole_camera);

#if 0
  // Prepare the bundle adjustment problem formulation .
  auto ba_problem = BundleAdjustmentProblem{};
  ba_problem.populate_data_from_two_view_geometry(
      feature_tracks, views.keypoints, match_index, two_view_geometry);


  // Solve the bundle adjustment problem with Ceres.
  ceres::Problem problem;
  for (int i = 0; i < ba_problem.observations.size(0); ++i)
  {
    auto cost_fn = ReprojectionError::create(ba_problem.observations(i, 0),
                                             ba_problem.observations(i, 1));

    problem.AddResidualBlock(cost_fn, nullptr /* squared loss */,
                             ba_problem.camera_parameters.data() +
                                 ba_problem.camera_indices[i] *
                                     CameraModelView<double>::dof(),
                             ba_problem.points_abs_coords_3d.data() +
                                 ba_problem.point_indices[i] * 3);
  }

  auto options = ceres::Solver::Options{};
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  auto summary = ceres::Solver::Summary{};
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  SARA_LOGI(logger, "Check the SfM...");
  SARA_DEBUG << "camera_parameters =\n"
             << ba_problem.camera_parameters.matrix() << std::endl;
  SARA_DEBUG << "points =\n"
             << ba_problem.points_abs_coords_3d.matrix() << std::endl;

  // TODO: check the point reprojection errors.
#endif

  return 0;
}
