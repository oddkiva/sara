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

#include "DO/Sara/Core/Pixel/SmartColorConversion.hpp"
#include <DO/Sara/SfM/BuildingBlocks/PointCloudManipulator.hpp>

#include <DO/Sara/Logging/Logger.hpp>

#include <unordered_set>


using namespace DO::Sara;


auto PointCloudManipulator::list_scene_point_indices(
    const FeatureTrack& track) const -> std::vector<ScenePointIndex>
{
  auto index_set = std::unordered_set<ScenePointIndex>{};
  for (const auto& v : track)
  {
    const auto scene_point_it = _from_vertex_to_scene_point_index.find(v);
    if (scene_point_it != _from_vertex_to_scene_point_index.end())
      index_set.emplace(scene_point_it->second);
  }

  const auto index_list = std::vector<ScenePointIndex>(  //
      index_set.begin(), index_set.end());

  return index_list;
}

auto PointCloudManipulator::filter_by_non_max_suppression(
    const FeatureTrack& track) const -> FeatureTrack
{
  struct VertexScorePair
  {
    FeatureVertex vertex;
    float score;
    auto operator<(const VertexScorePair& other) const -> bool
    {
      return score < other.score;
    }
  };

  auto filtered_set = std::unordered_map<PoseVertex, VertexScorePair>{};
  for (const auto& v : track)
  {
    const auto& f = feature(v);
    const auto& pose_vertex = _feature_graph[v].pose_vertex;
    const auto pose_vertex_it = filtered_set.find(pose_vertex);
    if (pose_vertex_it == filtered_set.end())
    {
      filtered_set[pose_vertex] = {.vertex = v, .score = f.extremum_value};
      continue;
    }

    auto& vertex_score = pose_vertex_it->second;
    if (vertex_score.score < f.extremum_value)
      vertex_score = {.vertex = v, .score = f.extremum_value};
  }

  auto filtered_list = FeatureTrack(filtered_set.size());
  std::transform(filtered_set.begin(), filtered_set.end(),
                 filtered_list.begin(),
                 [](const auto& v) { return v.second.vertex; });

  // Order feature vertices in a chronological order.
  //
  // The camera vertex ID is incremented as time goes on and can be seen as a
  // timestep.
  std::sort(filtered_list.begin(), filtered_list.end(),
            [this](const auto u, const auto v) {
              return _feature_graph[u].pose_vertex <
                     _feature_graph[v].pose_vertex;
            });

  return filtered_list;
}

auto PointCloudManipulator::find_feature_vertex_at_pose(
    const FeatureTrack& track,
    const PoseVertex pose_vertex) const -> std::optional<FeatureVertex>
{
  auto v = std::find_if(track.begin(), track.end(),
                        [this, pose_vertex](const auto& v) {
                          return this->gid(v).pose_vertex == pose_vertex;
                        });
  return v == track.end() ? std::nullopt : std::make_optional(*v);
}


auto PointCloudManipulator::barycenter(
    const std::vector<ScenePointIndex>& scene_point_indices) const -> ScenePoint
{
  if (scene_point_indices.empty())
    throw std::runtime_error{"Error: cannot calculate a barycentric scene "
                             "point from an empty list of scene point indices"};
  static const ScenePoint::Value zero = ScenePoint::Value::Zero();
  auto bary = std::accumulate(  //
      scene_point_indices.begin(), scene_point_indices.end(), zero,
      [this](const ScenePoint::Value& a,
             const ScenePointIndex bi) -> ScenePoint::Value {
        const ScenePoint::Value& b = _point_cloud[bi];
        return a + b;
      });
  bary /= scene_point_indices.size();

  auto scene_point = ScenePoint{};
  static_cast<ScenePoint::Value&>(scene_point) = bary;
  return scene_point;
}

auto PointCloudManipulator::split_by_scene_point_knowledge(
    const std::vector<FeatureTrack>& tracks) const
    -> std::pair<std::vector<FeatureTrack>, std::vector<FeatureTrack>>
{
  auto& logger = Logger::get();

  auto tracks_with_known_scene_point = std::vector<FeatureTrack>{};
  auto tracks_with_unknown_scene_point = std::vector<FeatureTrack>{};
  tracks_with_known_scene_point.reserve(tracks.size());
  tracks_with_unknown_scene_point.reserve(tracks.size());

  SARA_LOGD(logger, "Splitting feature tracks by knowledge of scene point...");

  for (const auto& track : tracks)
  {
    const auto scene_point_indices = list_scene_point_indices(track);
    if (scene_point_indices.empty())
      tracks_with_unknown_scene_point.emplace_back(track);
    else
      tracks_with_known_scene_point.emplace_back(track);
  }

  SARA_LOGD(logger, "Tracks: {}", tracks.size());
  SARA_LOGD(logger, "Tracks with known   scene point: {}",
            tracks_with_known_scene_point.size());
  SARA_LOGD(logger, "Tracks with unknown scene point: {}",
            tracks_with_unknown_scene_point.size());

  return std::make_pair(tracks_with_known_scene_point,
                        tracks_with_unknown_scene_point);
}

auto PointCloudManipulator::retrieve_scene_point_color(
    const Eigen::Vector3d& scene_point,  //
    const ImageView<Rgb8>& image,        //
    const QuaternionBasedPose<double>& pose,
    const v2::BrownConradyDistortionModel<double>& camera) const -> Rgb64f
{
  const auto& w = image.width();
  const auto& h = image.height();

  // Its coordinates in the camera frame.
  const auto camera_point = pose * scene_point;

  // Its corresponding pixel coordinates in the image.
  const Eigen::Vector2i u = camera
                                .project(camera_point)  //
                                .array()
                                .round()
                                .cast<int>();

  // Clamp for safety
  // TODO: do bilinear interpolation.
  const auto x = std::clamp(u.x(), 0, w - 1);
  const auto y = std::clamp(u.y(), 0, h - 1);

  // N.B.: the image is an array of BGR values.
  const auto& rgb8 = image(x, y);
  // We store RGB values.
  static constexpr auto normalization_factor = 1 / 255.;
  const Rgb64f rgb64f = rgb8.cast<double>() * normalization_factor;

  return rgb64f;
}


auto PointCloudManipulator::init_point_cloud(
    const std::vector<FeatureTrack>& feature_tracks,
    const ImageView<Rgb8>& image,  //
    const PoseEdge pose_edge,
    const v2::BrownConradyDistortionModel<double>& camera) -> void
{
  auto& logger = Logger::get();

  SARA_LOGD(logger, "Transform feature tracks into best feature pairs...");
  const auto& pose_u = boost::source(pose_edge, _pose_graph);
  const auto& pose_v = boost::target(pose_edge, _pose_graph);
  const auto& pose_data_u = _camera_pose_graph[pose_u].pose;
  const auto& pose_data_v = _camera_pose_graph[pose_v].pose;
  SARA_LOGD(logger, "Pose[from]:\n{}", pose_from.matrix34());
  SARA_LOGD(logger, "Pose[to  ]:\n{}", pose_to.matrix34());

#if 0
  const auto num_feature_tracks =
      static_cast<Eigen::Index>(feature_tracks.size());

  using FeatureVertexPair = std::array<FeatureVertex, 2>;
  auto best_feature_pairs = std::vector<FeatureVertexPair>(num_feature_tracks);
  std::transform(
      feature_tracks.begin(), feature_tracks.end(), best_feature_pairs.begin(),
      [this, camera_from, camera_to](const auto& track) -> FeatureVertexPair {
        // Non-max suppression.
        const auto track_filtered = filter_by_non_max_suppression(track);
        if (track_filtered.size() != 2)
          throw std::runtime_error{"Error: the feature track filtered by NMS "
                                   "is not a feature pair!"};

        // Retrieve the cleaned up feature correspondence.
        const auto v_from = find_vertex_at_camera_view(track_filtered,  //
                                                       camera_from);
        const auto v_to = find_vertex_at_camera_view(track_filtered,  //
                                                     camera_to);
        if (!v_from.has_value() || !v_to.has_value())
          throw std::runtime_error{
              "Error: the feature pair is not a valid feature correspondence!"};

        return {*v_from, *v_to};
      });

  SARA_LOGD(logger, "Calculating ray pairs from feature pairs...");
  auto rays_from = Eigen::MatrixXd{3, num_feature_tracks};
  auto rays_to = Eigen::MatrixXd{3, num_feature_tracks};
  for (auto t = 0u; t < num_feature_tracks; ++t)
  {
    const auto& feature_pair = best_feature_pairs[t];
    const auto coords = std::array{pixel_coords(feature_pair[0]),
                                   pixel_coords(feature_pair[1])};
    rays_from.col(t) = camera.backproject(coords[0]);
    rays_to.col(t) = camera.backproject(coords[1]);
  }

  // Calculate the associated triangulation.
  SARA_LOGD(logger, "Initialization the point cloud by 3D triangulation from "
                    "the relative pose...");
  const auto& relative_pose_attr = _pose_graph[pose_edge];
  if (!relative_pose_attr.relative_pose.has_value())
    throw std::runtime_error{
        "Error: tried triangulating but there is no relative pose!"};
  const auto& motion = *relative_pose_attr.relative_pose;
  if ((pose_to.matrix34() - motion.projection_matrix()).norm() > 1e-6)
    throw std::runtime_error{
        "Error: the absolute pose is not initialized from the relative pose!"};

  const auto triangulation = triangulate_linear_eigen(  //
      pose_from.matrix34(), pose_to.matrix34(),         //
      rays_from, rays_to);

  // Allocate the mapping from the feature vertices to the scene point index.
  if (!_from_vertex_to_scene_point_index.empty())
    _from_vertex_to_scene_point_index.clear();

  // Calculate the initial point cloud.
  if (!_point_cloud.empty())
    _point_cloud.clear();

  const auto& X = triangulation.X;
  const auto& scales_from = triangulation.scales[0];
  const auto& scales_to = triangulation.scales[1];

  auto scene_point_index = scene_point_index_t{};
  for (auto j = 0; j < X.cols(); ++j)
  {
    // Only consider **cheiral** inliers!
    if (!(scales_from(j) > 0 && scales_to(j) > 0))
      continue;

    const Eigen::Vector3d scene_coords = X.col(j).hnormalized();
    const auto rgb = retrieve_scene_point_color(scene_coords, image,  //
                                                pose_to, camera);
    const auto colored_point = (ColoredPoint{} << scene_coords, rgb).finished();

    // Add a new point to the point cloud.
    _point_cloud.emplace_back(colored_point);

    const auto& [u, v] = best_feature_pairs[j];

    // Assign a scene point index.
    _from_vertex_to_scene_point_index[u] = scene_point_index;
    _from_vertex_to_scene_point_index[v] = scene_point_index;
    ++scene_point_index;
  }

  SARA_LOGD(logger, "point cloud: {} 3D points", _point_cloud.size());
#endif
}
