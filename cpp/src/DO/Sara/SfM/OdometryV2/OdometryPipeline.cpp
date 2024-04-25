// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2024-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include "DO/Sara/MultiViewGeometry/Geometry/QuaternionBasedPose.hpp"
#include <DO/Sara/SfM/OdometryV2/OdometryPipeline.hpp>

#include <DO/Sara/Logging/Logger.hpp>

#include <DO/Sara/Core/Math/Rotation.hpp>
#include <DO/Sara/FeatureDetectors/SIFT.hpp>
#include <DO/Sara/Graphics/ImageDraw.hpp>
#include <DO/Sara/SfM/Helpers/KeypointMatching.hpp>
#include <DO/Sara/Visualization/Features/Draw.hpp>


using namespace DO::Sara;


auto v2::OdometryPipeline::set_config(
    const std::filesystem::path& video_path,
    const v2::BrownConradyDistortionModel<double>& camera) -> void
{
  // Build the dependency graph.
  _video_streamer.open(video_path);
  // The original camera.
  _camera = camera;
  // The virtual camera for the undistorted image.
  _camera_corrected.focal_lengths() << camera.fx(), camera.fy();
  _camera_corrected.shear() = camera.shear();
  _camera_corrected.principal_point() << camera.u0(), camera.v0();

  // Computer vision tasks.
  _distortion_corrector = std::make_unique<ImageDistortionCorrector>(
      _video_streamer.frame_rgb8(),     //
      _video_streamer.frame_gray32f(),  //
      _camera                           //
  );
  _rel_pose_estimator.configure(_camera);
  _point_cloud_generator = std::make_unique<PointCloudGenerator>(
      _pose_graph, _feature_tracker._feature_graph, _point_cloud);
}

auto v2::OdometryPipeline::read() -> bool
{
  return _video_streamer.read();
}

auto v2::OdometryPipeline::process() -> void
{
  if (_video_streamer.skip())
    return;

  auto& logger = Logger::get();
  SARA_LOGI(logger, "[Video Stream] Processing image frame {}",
            _video_streamer.frame_number());

  SARA_LOGI(logger, "[Image Distortion] Undistort image frame {}",
            _video_streamer.frame_number());
  _distortion_corrector->undistort();

  grow_geometry();
}

auto v2::OdometryPipeline::make_display_frame() const -> Image<Rgb8>
{
  return _distortion_corrector->frame_rgb8();
}

auto v2::OdometryPipeline::detect_keypoints(const ImageView<float>& image) const
    -> KeypointList<OERegion, float>
{
  auto& logger = Logger::get();
  const auto keys = compute_sift_keypoints(image,  //
                                           _feature_params.image_pyr_params);
  SARA_LOGI(logger, "[Keypoint Detection] {} keypoints", features(keys).size());
  return keys;
}

auto v2::OdometryPipeline::estimate_relative_pose(
    const KeypointList<OERegion, float>& keys_src,
    const KeypointList<OERegion, float>& keys_dst) const
    -> std::pair<RelativePoseData, TwoViewGeometry>
{
  auto& logger = Logger::get();

  if (features(keys_src).empty() || features(keys_dst).empty())
  {
    SARA_LOGI(logger, "[Relative Pose] Skipped image matching...");
    return {};
  }

  auto matches = match(keys_src, keys_dst, _feature_params.sift_nn_ratio);
  SARA_LOGI(logger, "[Relative Pose] Matched image keypoints...");
  if (matches.empty())
    return {};
  if (matches.size() > _feature_params.num_matches_max)
    matches.resize(_feature_params.num_matches_max);

  auto [two_view_geometry, inliers, sample_best] =
      _rel_pose_estimator.estimate_relative_pose(keys_src, keys_dst, matches);
  SARA_LOGI(logger, "[Relative Pose] Estimated relative pose...");

  const auto res = std::make_pair(  //
      RelativePoseData{.matches = std::move(matches),
                       .inliers = std::move(inliers),
                       .motion =
                           {
                               .R = two_view_geometry.C2.R,  //
                               .t = two_view_geometry.C2.t   //
                           }

      },
      std::move(two_view_geometry)  //
  );

  return res;
}

auto v2::OdometryPipeline::grow_geometry() -> bool
{
  auto& logger = Logger::get();

  // Detect and describe the local features.
  _pose_prev = _pose_curr;

  const auto frame_gray32f = _distortion_corrector->frame_gray32f();
  const auto frame_number = _video_streamer.frame_number();
  auto keys_curr = detect_keypoints(frame_gray32f);

  // Boundary case: the graphs are empty.
  if (_pose_graph.num_vertices() == 0)
  {
    // Initialize the new camera pose from the latest image frame.
    auto abs_pose_curr = QuaternionBasedPose<double>::identity();
    auto abs_pose_data = AbsolutePoseData{
        frame_number,             //
        std::move(keys_curr),     //
        std::move(abs_pose_curr)  //
    };
    _pose_curr = _pose_graph.add_absolute_pose(std::move(abs_pose_data));

    return true;
  }

  const auto& keys_prev = _pose_graph[_pose_prev].keypoints;
  auto [rel_pose_data, two_view_geometry] =
      estimate_relative_pose(keys_prev, keys_curr);
  const auto num_inliers = rel_pose_data.inliers.flat_array().count();
  SARA_LOGI(logger, "[SfM] Relative pose inliers: {} 3D points", num_inliers);
  if (num_inliers < _feature_params.num_inliers_min)
  {
    SARA_LOGI(logger, "[SfM] Relative pose failed!");
    return false;
  }
  SARA_LOGI(logger, "[SfM] Relative pose succeeded!");

  // 1. Add the absolute pose vertex.
  auto abs_pose_curr =
      _pose_graph.num_vertices() == 1
          ? QuaternionBasedPose<double>{}
          : QuaternionBasedPose<double>{
                .q = Eigen::Quaterniond{rel_pose_data.motion.R},
                .t = rel_pose_data.motion.t  //
            };
  auto abs_pose_data = AbsolutePoseData{
      frame_number,             //
      std::move(keys_curr),     //
      std::move(abs_pose_curr)  //
  };
  _pose_curr = _pose_graph.add_absolute_pose(std::move(abs_pose_data));

  // 2. Update the feature tracks by adding the feature matches that are
  //    verified by the relative pose estimation.
  //    Notice move semantics which will the relative pose data after this call.
  //
  // Note that in the case of only two views, feature tracks are "compressed"
  // feature matches.
  const auto pose_edge = _pose_graph.add_relative_pose(  //
      _pose_prev, _pose_curr,                            //
      std::move(rel_pose_data));
  _feature_tracker.update_feature_tracks(_pose_graph, pose_edge);

  // 3. Recalculate the feature tracks that are still alive.
  std::tie(_tracks_alive, _track_visibility_count) =
      _feature_tracker.calculate_alive_feature_tracks(_pose_curr);

  // Extra steps for when the pose graph contains 2 poses.
  if (_pose_graph.num_vertices() == 2)
  {
    _tracks_alive_without_scene_point = _tracks_alive;
  }
  // Extra steps for when the pose graph contains 3 poses or more.
  else
  {
    // 4. Propagate the scene point to the feature tracks that grew longer.
    //    The feature tracks that grew longer can only be those among the
    //    tracks that are still alive.
    SARA_LOGI(
        logger,
        "Propagating the scene points to feature tracks that grew longer...");
    _point_cloud_generator->propagate_scene_point_indices(_tracks_alive);

    // 5. Reassign a unique scene point cloud to each feature tracks by
    //    compressing the point cloud.
    SARA_LOGI(logger, "Compressing the point cloud...");
    _point_cloud_generator->compress_point_cloud(
        _feature_tracker._feature_tracks);

    // 6. Determine the current absolute pose from the alive tracks using a PnP
    //    approach.
    std::tie(_tracks_alive_with_known_scene_point,
             _tracks_alive_without_scene_point) =
        _point_cloud_generator->split_by_scene_point_knowledge(_tracks_alive);
    const auto [abs_pose_mat, abs_pose_est_successful] =
        _abs_pose_estimator.estimate_pose(_tracks_alive_with_known_scene_point,
                                          _pose_curr, _camera_corrected,
                                          *_point_cloud_generator);
    if (!abs_pose_est_successful)
      return false;

    // 7. Update the current absolute pose, which was initialized dummily.
    abs_pose_curr = QuaternionBasedPose<double>{
        Eigen::Quaterniond{abs_pose_mat.leftCols<3>()},  //
        abs_pose_mat.col(3)                              //
    };
    _pose_graph[_pose_curr].pose = abs_pose_curr;
  }

  // 8. Grow point cloud by triangulation.
  //
  // TODO: don't add 3D scene points that are too far, like points in the sky
  const auto frame_corrected = _distortion_corrector->frame_rgb8();
  _point_cloud_generator->grow_point_cloud(_tracks_alive_without_scene_point,
                                           frame_corrected, pose_edge,
                                           _camera_corrected);

  // The rotation is expressed in the camera coordinates.
  // But the calculation is done in the automotive/aeronautics coordinate
  // system.
  //
  // The z-coordinate of the camera coordinates is the x-axis of the
  // automotive coordinates
  //
  // clang-format off
  static const auto P = (Eigen::Matrix3d{} <<
     0,  0, 1,
    -1,  0, 0,
     0, -1, 0
  ).finished();
  // clang-format on

  const auto& R = _pose_graph[pose_edge].motion.R;
  const Eigen::Matrix3d R_delta_abs = P * R.transpose() * P.transpose();
  _current_global_rotation = R_delta_abs * _current_global_rotation;

  const auto q_global = Eigen::Quaterniond{_current_global_rotation};
  auto angles = calculate_yaw_pitch_roll(q_global);
  static constexpr auto degrees = 180. / M_PI;
  SARA_LOGI(logger, "Global yaw   = {} deg", angles(0) * degrees);
  SARA_LOGI(logger, "Global pitch = {} deg", angles(1) * degrees);
  SARA_LOGI(logger, "Global roll  = {} deg", angles(2) * degrees);

  return true;
}
