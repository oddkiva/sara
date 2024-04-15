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

#include <DO/Sara/SfM/OdometryV2/OdometryPipeline.hpp>

#include <DO/Sara/Logging/Logger.hpp>

#include <DO/Sara/Graphics/ImageDraw.hpp>
#include <DO/Sara/Visualization/Features/Draw.hpp>

#include <DO/Sara/FeatureDetectors/SIFT.hpp>
#include <DO/Sara/SfM/Helpers/KeypointMatching.hpp>


using namespace DO::Sara;


auto v2::OdometryPipeline::set_config(
    const std::filesystem::path& video_path,
    const v2::BrownConradyDistortionModel<double>& camera) -> void
{
  // Build the dependency graph.
  _video_streamer.open(video_path);
  _camera = camera;

  // Computer vision tasks.
  _distortion_corrector = std::make_unique<ImageDistortionCorrector>(
      _video_streamer.frame_rgb8(),     //
      _video_streamer.frame_gray32f(),  //
      _camera                           //
  );
  _relative_pose_estimator.configure(_camera);
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

  _distortion_corrector->undistort();

  add_camera_pose();
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
  SARA_LOGI(logger, "[Feature Detection] {} keypoints", features(keys).size());
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
      _relative_pose_estimator.estimate_relative_pose(keys_src, keys_dst,
                                                      matches);
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

auto v2::OdometryPipeline::add_camera_pose() -> bool
{
  auto& logger = Logger::get();

  // Detect and describe the local features.
  _pose_prev = _pose_curr;

  const auto frame = _distortion_corrector->frame_gray32f();
  const auto frame_number = _video_streamer.frame_number();
  auto keys_curr = detect_keypoints(frame);

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

  if (_pose_graph.num_vertices() == 1)
  {
    auto abs_pose_curr = QuaternionBasedPose<double>{
        .q = Eigen::Quaterniond{rel_pose_data.motion.R},
        .t = rel_pose_data.motion.t  //
    };

    auto abs_pose_data = AbsolutePoseData{
        frame_number,             //
        std::move(keys_curr),     //
        std::move(abs_pose_curr)  //
    };

    // 1. Add the absolute pose vertex.
    _pose_graph.add_absolute_pose(std::move(abs_pose_data));

    // 2. Add the pose edge, which will invalidate the relative pose data.
    const auto pose_edge = _pose_graph.add_relative_pose(
        _pose_prev, _pose_curr, std::move(rel_pose_data));

    // 3. Grow the feature graph by adding the feature matches.
    _feature_tracker.update_feature_tracks(_pose_graph, pose_edge);
    std::tie(_tracks_alive, _track_visibility_count) =
        _feature_tracker.calculate_alive_feature_tracks(_pose_curr);

    // 4. TODO: Init point cloud

    // 5. TODO: don't add 3D scene points that are too far, like point in the
    //    sky

    return true;
  }

  // 1. Grow the feature graph first by adding the feature matches that are
  //    deemed reliable from the relative pose estimation.
  // 2. Recalculate the feature tracks.
  // 3. Get the feature tracks that are still alive.
  // 4. For each feature track still alive, get the corresponding scene
  //    points.
  //    Each alive feature track still has the same old feature IDs in the
  //    previous image frames, and we know their scene points.
  //    Use triangulation computer vision task, to calculate the new camera
  //    absolute pose.
  // 5. With the camera absolute pose, add the new scene points.
  //    Specifically, they are the alive feature tracks (with cardinality 2)
  //    for which we don't know the scene points yet.

  // TODO: Grow point cloud by triangulation.
  return false;
}
