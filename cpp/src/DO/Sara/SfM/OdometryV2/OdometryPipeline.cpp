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
}

auto v2::OdometryPipeline::make_display_frame() const -> Image<Rgb8>
{
  return _distortion_corrector->frame_rgb8();
}

auto v2::OdometryPipeline::detect_keypoints(const ImageView<float>& image) const
    -> KeypointList<OERegion, float>
{
  return compute_sift_keypoints(image, _feature_params.image_pyr_params);
}

auto v2::OdometryPipeline::estimate_relative_pose(
    const CameraPoseGraph::Vertex pose_u,  //
    const CameraPoseGraph::Vertex pose_v) const
    -> std::pair<RelativePoseData, TwoViewGeometry>
{
  auto& logger = Logger::get();

  SARA_LOGI(logger, "Matching features...");
  const auto& keys_u = _pose_graph[pose_u].keypoints;
  const auto& keys_v = _pose_graph[pose_v].keypoints;
  if (features(keys_u).empty() || features(keys_v).empty())
    return {};

  auto matches = match(keys_u, keys_v, _feature_params.sift_nn_ratio);
  if (matches.empty())
    return {};
  if (matches.size() > _feature_params.num_matches_max)
    matches.resize(_feature_params.num_matches_max);

  SARA_LOGI(logger, "Estimating relative pose...");
  auto [geometry, inliers, sample_best] =
      _relative_pose_estimator.estimate_relative_pose(keys_u, keys_v, matches);
  const auto num_inliers = inliers.flat_array().count();
  SARA_LOGI(logger, "inlier count: {}", num_inliers);

  return {
      RelativePoseData{.matches = std::move(matches),
                       .inliers = std::move(inliers),
                       .motion = {}

      },
      geometry  //
  };
}

auto v2::OdometryPipeline::add_camera_pose_and_grow_point_cloud() -> bool
{
  auto& logger = Logger::get();

  // Detect and describe the local features.
  _pose_prev = _pose_curr;
  const auto frame = _distortion_corrector->frame_gray32f();
  const auto frame_number = _video_streamer.frame_number();
  auto keypoints = detect_keypoints(frame);
  _pose_curr = _pose_graph.add_absolute_pose(std::move(keypoints),  //
                                             frame_number);

  const auto& pose_data = _pose_graph[_pose_curr];
  SARA_LOGI(logger, "Camera [frame:{}]: {} keypoints",  //
            pose_data.image_id, features(pose_data.keypoints).size());

  // We need two frames at least for the epipolar geometry.
  if (_pose_graph.num_vertices() < 2)
    return false;

  return false;
}
