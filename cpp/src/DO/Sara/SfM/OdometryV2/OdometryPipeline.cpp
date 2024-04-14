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
  SARA_LOGI(logger, "[Feature Detection] Matching image keypoints...");
  return compute_sift_keypoints(image, _feature_params.image_pyr_params);
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

  if (_pose_graph.num_vertices() == 1)
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
  else
  {
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

    if (_pose_graph.num_vertices() == 2)
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
      _pose_graph.add_relative_pose(_pose_prev, _pose_curr,
                                    std::move(rel_pose_data));

      // 3. TODO: Init point cloud

      return true;
    }
    else
    {
      // 1. Add the absolute pose vertex.

      // TODO: Grow point cloud by triangulation.
      return false;
    }
  }

  return false;
}
