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
  auto& logger = Logger::get();
  SARA_LOGI(logger, "[Feature Detection] Matching image keypoints...");
  return compute_sift_keypoints(image, _feature_params.image_pyr_params);
}

auto v2::OdometryPipeline::estimate_relative_pose(
    const CameraPoseGraph::Vertex pose_u,  //
    const CameraPoseGraph::Vertex pose_v) const
    -> std::pair<RelativePoseData, TwoViewGeometry>
{
  auto& logger = Logger::get();

  const auto& keys_u = _pose_graph[pose_u].keypoints;
  const auto& keys_v = _pose_graph[pose_v].keypoints;
  if (features(keys_u).empty() || features(keys_v).empty())
  {
    SARA_LOGI(logger, "[Relative Pose] Skipped image matching...");
    return {};
  }

  auto matches = match(keys_u, keys_v, _feature_params.sift_nn_ratio);
  SARA_LOGI(logger, "[Relative Pose] Matched image keypoints...");
  if (matches.empty())
    return {};
  if (matches.size() > _feature_params.num_matches_max)
    matches.resize(_feature_params.num_matches_max);

  auto [two_view_geometry, inliers, sample_best] =
      _relative_pose_estimator.estimate_relative_pose(keys_u, keys_v, matches);
  SARA_LOGI(logger, "[Relative Pose] Estimated relative pose...");

  const auto res = std::pair{
      RelativePoseData{.matches = std::move(matches),
                       .inliers = std::move(inliers),
                       .motion =
                           {
                               .R = two_view_geometry.C2.R,  //
                               .t = two_view_geometry.C2.t   //
                           }

      },
      two_view_geometry  //
  };

  return res;
}

auto v2::OdometryPipeline::update_absolute_pose_from_latest_relative_pose_data(
    const RelativePoseData& relative_pose_data,
    const TwoViewGeometry& two_view_geometry) -> bool
{
  auto& logger = Logger::get();

  const auto num_inliers = relative_pose_data.inliers.flat_array().count();
  SARA_LOGI(logger, "[SfM] Relative pose inliers: {} 3D points", num_inliers);
  if (num_inliers < _feature_params.num_inliers_min)
  {
    SARA_LOGI(logger, "[SfM] Relative pose failed!");
    return false;
  }
  SARA_LOGI(logger, "[SfM] Relative pose succeeded!");

  if (_pose_graph.num_vertices() == 2)
  {
    SARA_LOGI(logger, "Initializing the first two camera poses...");
    // Set the absolute pose of the first camera which is the identity rigid
    // body transformation.
    auto& initial_pose = _pose_graph[_pose_prev].pose;
    {
      initial_pose.q.setIdentity();
      initial_pose.t.setZero();
    }

    // Set the absolute pose of the first camera as the first relative pose.
    auto& second_pose = _pose_graph[_pose_curr].pose;
    {
      // HEURISTICS: Advance from only 5 cm at most to view something nice.
      const auto [e, edge_exists] = _pose_graph.edge(_pose_prev, _pose_curr);
      if (!edge_exists)
        throw std::runtime_error{"Edge must exist!"};

      _pose_graph[e] = relative_pose_data;
    }

    // TODO: make a new function for this.
    // SARA_LOGI(logger, "Initializing the point cloud...");
    // _point_cloud_operator->init_point_cloud(_tracks_alive, current_image,
    //                                         _relative_pose_edge_id, _camera);
  }
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
  SARA_LOGI(logger,
            "[SfM] Initialized new camera pose[frame:{}]: {} keypoints",  //
            pose_data.image_id, features(pose_data.keypoints).size());

  // We need two frames at least for the epipolar geometry.
  if (_pose_graph.num_vertices() < 2)
    return false;

  const auto [relative_pose_data, two_view_geometry] =
      this->estimate_relative_pose(_pose_prev, _pose_curr);
  const auto num_inliers = relative_pose_data.inliers.flat_array().count();
  SARA_LOGI(logger, "[SfM] Relative pose inliers: {} 3D points", num_inliers);
  if (num_inliers < _feature_params.num_inliers_min)
  {
    SARA_LOGI(logger, "[SfM] Relative pose failed!");
    return false;
  }
  SARA_LOGI(logger, "[SfM] Relative pose succeeded!");

  if (_pose_graph.num_vertices() == 2)
  {
    // Init point cloud.
  }
  else
  {
    // Grow point cloud by triangulation.
  }


  return false;
}
