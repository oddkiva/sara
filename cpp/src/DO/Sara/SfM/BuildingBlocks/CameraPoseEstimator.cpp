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

#include <DO/Sara/SfM/BuildingBlocks/CameraPoseEstimator.hpp>

#include <DO/Sara/Logging/Logger.hpp>
#include <DO/Sara/RANSAC/RANSACv2.hpp>


using namespace DO::Sara;


auto CameraPoseEstimator::estimate_pose(
    const PointRayCorrespondenceList<double>& point_ray_pairs,
    const CameraIntrinsicModel& camera)
    -> std::tuple<PoseMatrix, Inlier, MinimalSample>
{
  _inlier_predicate.set_camera(camera);

  static constexpr auto debug = false;
  return v2::ransac(point_ray_pairs, _solver, _inlier_predicate,
                    _ransac_iter_max, _ransac_confidence_min, std::nullopt,
                    debug);
}

auto CameraPoseEstimator::estimate_pose(
    const std::vector<FeatureTrack>& valid_ftracks,
    const CameraPoseGraph::Vertex pv,
    const CameraPoseEstimator::CameraIntrinsicModel& camera,
    const PointCloudGenerator& pcg) -> std::pair<PoseMatrix, bool>
{
  auto& logger = Logger::get();

  const auto num_ftracks = static_cast<Eigen::Index>(valid_ftracks.size());

  SARA_LOGD(logger, "Applying NMS to the {} feature tracks...", num_ftracks);

  auto ftracks_filtered = std::vector<FeatureTrack>{};
  ftracks_filtered.resize(num_ftracks);
  std::transform(
      valid_ftracks.begin(), valid_ftracks.end(), ftracks_filtered.begin(),
      [&pcg, pv](const FeatureTrack& ftrack) -> FeatureTrack {
        const auto ftrack_filtered = pcg.filter_by_non_max_suppression(ftrack);
        const auto fvertex = pcg.find_feature_vertex_at_pose(  //
            ftrack_filtered, pv);
        if (!fvertex.has_value())
          throw std::runtime_error{"Error: the filtered track must contain the "
                                   "target camera vertex!"};
        if (ftrack_filtered.size() <= 2)
          throw std::runtime_error{"Error: a filtered feature track can't "
                                   "possibly have cardinality 2!"};
        return ftrack_filtered;
      });

  auto point_ray_pairs = PointRayCorrespondenceList<double>{};
  point_ray_pairs.x.resize({num_ftracks, 3});
  point_ray_pairs.y.resize({num_ftracks, 3});

  // Data collection.
  //
  // 1. Collect the scene point coordinates.
  SARA_LOGD(logger, "Retrieving scene points for each feature track...");
  auto scene_coords = point_ray_pairs.x.colmajor_view().matrix();
  for (auto t = 0; t < num_ftracks; ++t)
  {
    const auto& ftrack = ftracks_filtered[t];

    // Fetch the 3D scene coordinates.
    const auto scene_point_indices = pcg.list_scene_point_indices(ftrack);
    if (scene_point_indices.empty())
      throw std::runtime_error{
          "Error: a feature track must be assigned a scene point index!"};

    // If there are more than one scene point index, we fetch the barycentric
    // coordinates anyway.
    scene_coords.col(t) = pcg.barycenter(scene_point_indices).coords();
    if (t < 10)
      std::cout << t << " -> " << scene_coords.col(t).transpose() << std::endl;
  }

  // 2. Collect the backprojected rays from the current camera view for each
  //    feature track.
  SARA_LOGD(logger,
            "Calculating backprojected rays from camera pose [{}] for each "
            "feature track...",
            pv);
  auto rays = point_ray_pairs.y.colmajor_view().matrix();
  for (auto t = 0; t < num_ftracks; ++t)
  {
    const auto& ftrack = ftracks_filtered[t];
    // Calculate the backprojected ray.
    const auto& fv = pcg.find_feature_vertex_at_pose(ftrack, pv);
    if (!fv.has_value())
      throw std::runtime_error{"Error: the feature track must be alive!"};
    const auto pixel_coords = pcg.pixel_coords(*fv).cast<double>();
    // Normalize the rays. This is important for Lambda-Twist P3P method.
    rays.col(t) = camera.backproject(pixel_coords).normalized();
    if (t < 10)
      SARA_LOGD(logger, "Backproject point {}:\n{} -> {}", t,
                pixel_coords.transpose().eval(),
                rays.col(t).transpose().eval());
  }

  // 3. solve the PnP problem with RANSAC.
  const auto [pose, inliers, sample_best] =
      estimate_pose(point_ray_pairs, camera);
  SARA_LOGD(logger, "[AbsPoseEst] Pose:\n{}", pose);
  SARA_LOGD(logger, "[AbsPoseEst] inlier count: {}",
            inliers.flat_array().count());
  const auto pose_estimated_successfully =
      inliers.flat_array().count() >= _ransac_inliers_min;

  return {pose, pose_estimated_successfully};
}
