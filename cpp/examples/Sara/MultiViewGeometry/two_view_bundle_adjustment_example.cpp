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
#include <DO/Sara/MultiViewGeometry/Camera/v2/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>
#include <DO/Sara/SfM/BuildingBlocks/BAReprojectionError.hpp>
#include <DO/Sara/SfM/BuildingBlocks/BundleAdjuster.hpp>
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


auto calculate_yaw_pitch_roll_angles(const Eigen::Vector3d& angle_axis_3d,
                                     const bool in_degrees = true)
    -> Eigen::Vector3d
{
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

  static constexpr auto degree = 180. / M_PI;

  const auto angle = angle_axis_3d.norm();
  const auto axis = Eigen::Vector3d{angle_axis_3d.normalized()};
  const auto R = Eigen::AngleAxisd{angle, axis}  //
                     .toRotationMatrix();
  const Eigen::Matrix3d Rw = P * R.transpose() * P.transpose();
  const auto angles = sara::calculate_yaw_pitch_roll(Eigen::Quaterniond{Rw});

  return in_degrees ? angles * degree : angles;
}

GRAPHICS_MAIN()
{
  auto& logger = sara::Logger::get();

  // Load the image data.
  SARA_LOGI(logger, "Loading images...");
  const auto data_dir_path = fs::path{
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
      sara::imread<sara::Rgb8>(image_paths[0].string()),  //
      sara::imread<sara::Rgb8>(image_paths[1].string())   //
  };

  // Load the calibration matrices.
  SARA_LOGI(logger, "Loading the internal camera matrices...");
  const auto K = std::vector{
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
      }  //
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
          }  //
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
      feature_tracker.find_feature_tracks_at_pose(1);

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

  auto ba = sara::BundleAdjuster{};
  static constexpr auto intrinsics_dim = 4;  // fx, fy, u0, v0
  static constexpr auto extrinsics_dim = 6;
  ba.form_problem(pose_graph, feature_tracker, K, point_cloud_generator,
                  tracks_filtered, intrinsics_dim, extrinsics_dim);
  // Freeze all the intrinsic parameters during the optimization.
  SARA_LOGI(logger, "Freezing intrinsic camera parameters...");
  for (auto v = 0; v < 2; ++v)
    ba.problem->SetParameterBlockConstant(ba.data.intrinsics[v].data());

  // Freeze the first absolute pose parameters.
  SARA_LOGI(logger, "Freezing first absolute pose...");
  ba.problem->SetParameterBlockConstant(ba.data.extrinsics[0].data());

  const auto& ba_data = ba.data;
  SARA_LOGI(
      logger, "[BA][BEFORE] points =\n{}",
      Eigen::MatrixXd{ba_data.point_coords.matrix().topRows<20>().eval()});
  {
    const auto angles = calculate_yaw_pitch_roll_angles(
        ba_data.extrinsics.matrix().row(1).head(3).transpose());

    SARA_LOGI(logger, "[BEFORE] yaw   = {} deg", angles(0));
    SARA_LOGI(logger, "[BEFORE] pitch = {} deg", angles(1));
    SARA_LOGI(logger, "[BEFORE] roll  = {} deg", angles(2));
  }

  // Solve the BA.
  ba.solve();

  SARA_LOGI(logger, "Checking the BA...");
  SARA_LOGI(logger, "[BA][AFTER ] camera_parameters =\n{}",
            Eigen::MatrixXd{ba_data.extrinsics.matrix()});
  SARA_LOGI(logger, "[BA][AFTER ] points =\n{}",
            Eigen::MatrixXd{ba_data.point_coords.matrix().topRows<20>()});

  {
    const auto angles = calculate_yaw_pitch_roll_angles(
        ba_data.extrinsics.matrix().row(1).head(3).transpose());
    SARA_LOGI(logger, "[AFTER ] yaw   = {} deg", angles(0));
    SARA_LOGI(logger, "[AFTER ] pitch = {} deg", angles(1));
    SARA_LOGI(logger, "[AFTER ] roll  = {} deg", angles(2));
  }


  return 0;
}
