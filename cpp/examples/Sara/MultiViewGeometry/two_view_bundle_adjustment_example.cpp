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

#include <DO/Sara/Core/Math/AxisConvention.hpp>
#include <DO/Sara/Core/Math/Rotation.hpp>
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

#include <algorithm>
#include <filesystem>


using namespace std;
using namespace std::string_literals;

namespace fs = std::filesystem;
namespace sara = DO::Sara;


struct ReprojectionError
{
  static constexpr auto ResidualDimension = 2;
  static constexpr auto IntrinsicParameterCount = 4;
  static constexpr auto ExtrinsicParameterCount = 6;
  static constexpr auto PointDimension = 3;

  ReprojectionError(double observed_x, double observed_y)
    : observed_x{observed_x}
    , observed_y{observed_y}
  {
  }

  template <typename T>
  bool operator()(const T* const extrinsics,  // (1) extrinsic camera parameters
                  const T* const intrinsics,  // (2) intrinsic camera parameters
                  const T* const point,       // (3) 3D point
                  T* residuals) const
  {
    T p[3];

    // Rotate the point.
    ceres::AngleAxisRotatePoint(extrinsics, point, p);
    // Translate the point.
    const auto t = extrinsics + 3;
    p[0] += t[0];
    p[1] += t[1];
    p[2] += t[2];

    // Normalized camera coordinates.
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    // Apply the internal parameters.
    const auto& fx = intrinsics[0];
    const auto& fy = intrinsics[1];
    const auto& u0 = intrinsics[2];
    const auto& v0 = intrinsics[3];
    const auto predicted_x = fx * xp + u0;
    const auto predicted_y = fy * yp + v0;

    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  static ceres::CostFunction* create(const double observed_x,
                                     const double observed_y)
  {
    return new ceres::AutoDiffCostFunction<
        ReprojectionError, ResidualDimension,  //
        ExtrinsicParameterCount, IntrinsicParameterCount, PointDimension>{
        new ReprojectionError{observed_x, observed_y}  //
    };
  }

  double observed_x;
  double observed_y;
};


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
  if ((K[0] - K[1]).norm() > 1e-8)
    throw std::runtime_error{
        "We require identical camera matrices in for now!"};
  for (auto i = 0; i < 2; ++i)
    SARA_LOGD(logger, "K[{}] =\n{}", i, K[i]);

  SARA_LOGI(logger, "Computing keypoints...");
  const auto image_pyr_params = sara::ImagePyramidParams(0);
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
  const auto [tracks, track_visibility_count] =
      feature_tracker.calculate_alive_feature_tracks(1);

  auto point_cloud = sara::PointCloudGenerator::PointCloud{};
  auto point_cloud_generator = sara::PointCloudGenerator{
      pose_graph, feature_tracker._feature_graph, point_cloud};

  auto pinhole_camera = sara::v2::PinholeCamera<double>{};
  pinhole_camera.fx() = K[1](0, 0);
  pinhole_camera.fy() = K[1](1, 1);
  pinhole_camera.shear() = K[1](0, 1);
  pinhole_camera.u0() = K[1](0, 2);
  pinhole_camera.v0() = K[1](1, 2);
  point_cloud_generator.grow_point_cloud(tracks, images[1], pose_edge,
                                         pinhole_camera);

  // Filter the feature tracks by NMS.
  auto tracks_filtered = std::vector<sara::FeatureTracker::Track>{};
  for (const auto& track : tracks)
  {
    // Does a track have a 3D point? If not, discard it.
    const auto p = point_cloud_generator.scene_point(track.front());
    if (p == std::nullopt)
      continue;

    // Filter the feature track by NMS: there should be only 1 feature per
    // image.
    auto track_filtered = point_cloud_generator  //
                              .filter_by_non_max_suppression(track);

    tracks_filtered.emplace_back(std::move(track_filtered));
  }

  // Collect the BA data.
  const auto num_scene_points = static_cast<int>(tracks_filtered.size());
  auto num_image_points = 0;
  for (const auto& track : tracks_filtered)
    num_image_points += static_cast<int>(track.size());

  auto ba_data = sara::BundleAdjustmentData{};
  static constexpr auto num_views = 2;
  static constexpr auto num_intrinsics = 4;  // fx, fy, u0, v0
  static constexpr auto num_extrinsics = 6;
  ba_data.resize(num_image_points, num_scene_points, num_views,  //
                 num_intrinsics, num_extrinsics);

  SARA_LOGI(logger, "Populating the BA observation/image point data...");
  auto o = 0;  // observation index.
  for (auto t = std::size_t{}; t < tracks_filtered.size(); ++t)
  {
    const auto& track = tracks_filtered[t];
    for (const auto& u : track)
    {
      const Eigen::Vector2d pixel_coords = point_cloud_generator  //
                                               .pixel_coords(u)
                                               .cast<double>();
      ba_data.observations(o, 0) = pixel_coords.x();
      ba_data.observations(o, 1) = pixel_coords.y();

      ba_data.point_indices[o] = static_cast<int>(t);
      ba_data.camera_indices[o] =
          static_cast<int>(feature_tracker._feature_graph[u].pose_vertex);

      ++o;
    }
  }

  SARA_LOGI(logger, "Populating the BA (3D) point data...");
  for (auto t = std::size_t{}; t < tracks_filtered.size(); ++t)
  {
    // Retrieve the scene point.
    const auto& track = tracks_filtered[t];
    const auto scene_point = point_cloud_generator.scene_point(track.front());

    // Store.
    const auto tt = static_cast<int>(t);
    ba_data.point_coords[tt].vector() = scene_point->coords();
  }

  SARA_LOGI(logger, "Populating the BA camera parameter data...");
  auto extrinsics_params = ba_data.extrinsics.matrix();
  auto intrinsics_params = ba_data.intrinsics.matrix();
  SARA_LOGI(logger, "Filling parameters...");
  using PoseVertex = sara::CameraPoseGraph::Vertex;
  for (auto v = PoseVertex{}; v < pose_graph.num_vertices(); ++v)
  {
    // Angle axis vector.
    auto extrinsics_v = extrinsics_params.row(v);
    auto intrinsics_v = intrinsics_params.row(v);

    // The original data.
    const auto& pose_v = pose_graph[v].pose;
    const auto aaxis_v = Eigen::AngleAxisd{pose_v.q};
    const Eigen::Vector3d aaxis_v_3d = aaxis_v.angle() * aaxis_v.axis();
    // Initialize the absolute rotation.
    extrinsics_v << aaxis_v_3d.transpose(), pose_v.t.transpose();

    std::cout << extrinsics_v << std::endl;
    SARA_LOGD(logger, "Populating extrinsics[{}]=\n{}", v, extrinsics_v.eval());

    // Initialize the internal camera parameters.
    intrinsics_v(0) = K[v](0, 0);  // fx
    intrinsics_v(1) = K[v](1, 1);  // fy
    intrinsics_v(2) = K[v](0, 2);  // u0
    intrinsics_v(3) = K[v](1, 2);  // v0
    SARA_LOGD(logger, "Populating intrinsics[{}]=\n{}", v, intrinsics_v.eval());
  }

  // Solve the bundle adjustment problem with Ceres.
  SARA_LOGI(logger, "Forming the BA problem...");
  auto ba_problem = ceres::Problem{};
  for (auto i = 0; i < num_image_points; ++i)
  {
    SARA_LOGT(logger, "Adding residual with image point {}...", i);

    // Create a cost residual function.
    const auto cost_fn = ReprojectionError::create(ba_data.observations(i, 0),
                                                   ba_data.observations(i, 1));

    // Locate the parameter data.
    const auto camera_idx = ba_data.camera_indices[i];
    const auto point_idx = ba_data.point_indices[i];
    const auto extrinsics_ptr = ba_data.extrinsics[camera_idx].data();
    const auto intrinsics_ptr = ba_data.intrinsics[camera_idx].data();
    const auto scene_point_ptr = ba_data.point_coords[point_idx].data();

    ba_problem.AddResidualBlock(cost_fn, nullptr /* squared loss */,  //
                                extrinsics_ptr, intrinsics_ptr,
                                scene_point_ptr);
  }

  // Freeze all the intrinsic parameters during the optimization.
  SARA_LOGI(logger, "Freezing intrinsic camera parameters...");
  for (auto v = 0; v < 2; ++v)
    ba_problem.SetParameterBlockConstant(ba_data.intrinsics[v].data());

  // Freeze the first absolute pose parameters.
  SARA_LOGI(logger, "Freezing first absolute pose...");
  ba_problem.SetParameterBlockConstant(ba_data.extrinsics[0].data());

  SARA_LOGI(logger, "[BA][BEFORE] camera_parameters =\n{}",
            Eigen::MatrixXd{ba_data.extrinsics.matrix()});
  SARA_LOGI(
      logger, "[BA][BEFORE] points =\n{}",
      Eigen::MatrixXd{ba_data.point_coords.matrix().topRows<20>().eval()});
  // The rotation is expressed in the camera coordinates.
  // But the calculation is done in the automotive/aeronautics coordinate
  // system.
  //
  // The z-coordinate of the camera coordinates is the x-axis of the automotive
  // coordinates
  static const auto P =
      sara::axis_permutation_matrix(sara::AxisConvention::Automotive)
          .cast<double>()
          .eval();
  {
    const auto aaxis_1 = ba_data.extrinsics.matrix().row(1).head(3);
    const auto angle_1 = aaxis_1.norm();
    const auto axis_1 = Eigen::Vector3d{aaxis_1.normalized()};
    const auto R_1 = Eigen::AngleAxisd{angle_1, axis_1}  //
                         .toRotationMatrix();
    const Eigen::Matrix3d Rw_1 = P * R_1.transpose() * P.transpose();
    const auto angles =
        sara::calculate_yaw_pitch_roll(Eigen::Quaterniond{Rw_1});
    static constexpr auto degree = 180. / M_PI;
    SARA_LOGI(logger, "[BEFORE] yaw   = {} deg", angles(0) * degree);
    SARA_LOGI(logger, "[BEFORE] pitch = {} deg", angles(1) * degree);
    SARA_LOGI(logger, "[BEFORE] roll  = {} deg", angles(2) * degree);
  }

  SARA_LOGI(logger, "[BA]Solving the BA problem...");
  auto options = ceres::Solver::Options{};
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  auto summary = ceres::Solver::Summary{};
  ceres::Solve(options, &ba_problem, &summary);
  SARA_LOGI(logger, "{}", summary.BriefReport());

  SARA_LOGI(logger, "Checking the BA...");
  SARA_LOGI(logger, "[BA][AFTER ] camera_parameters =\n{}",
            Eigen::MatrixXd{ba_data.extrinsics.matrix()});
  SARA_LOGI(logger, "[BA][AFTER ] points =\n{}",
            Eigen::MatrixXd{ba_data.point_coords.matrix().topRows<20>()});

  {
    const auto aaxis_1 = ba_data.extrinsics.matrix().row(1).head(3);
    const auto angle_1 = aaxis_1.norm();
    const auto axis_1 = Eigen::Vector3d{aaxis_1.normalized()};
    const auto R_1 = Eigen::AngleAxisd{angle_1, axis_1}  //
                         .toRotationMatrix();
    const Eigen::Matrix3d Rw_1 = P * R_1.transpose() * P.transpose();
    const auto angles =
        sara::calculate_yaw_pitch_roll(Eigen::Quaterniond{Rw_1});
    static constexpr auto degree = 180. / M_PI;
    SARA_LOGI(logger, "[AFTER ] yaw   = {} deg", angles(0) * degree);
    SARA_LOGI(logger, "[AFTER ] pitch = {} deg", angles(1) * degree);
    SARA_LOGI(logger, "[AFTER ] roll  = {} deg", angles(2) * degree);
  }


  return 0;
}
