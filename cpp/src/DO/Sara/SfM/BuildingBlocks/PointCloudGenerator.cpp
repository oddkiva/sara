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

#include <DO/Sara/SfM/BuildingBlocks/PointCloudGenerator.hpp>

#include <DO/Sara/Logging/Logger.hpp>


using namespace DO::Sara;


auto PointCloudGenerator::list_scene_point_indices(
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

auto PointCloudGenerator::filter_by_non_max_suppression(
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

auto PointCloudGenerator::find_feature_vertex_at_pose(
    const FeatureTrack& track, const PoseVertex pose_vertex) const
    -> std::optional<FeatureVertex>
{
  auto v = std::find_if(track.begin(), track.end(),
                        [this, pose_vertex](const auto& v) {
                          return this->gid(v).pose_vertex == pose_vertex;
                        });
  return v == track.end() ? std::nullopt : std::make_optional(*v);
}


auto PointCloudGenerator::barycenter(
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

  return bary;
}

auto PointCloudGenerator::split_by_scene_point_knowledge(
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

auto PointCloudGenerator::retrieve_scene_point_color(
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


auto PointCloudGenerator::seed_point_cloud(
    const std::vector<FeatureTrack>& tracks,
    const ImageView<Rgb8>& image,  //
    const PoseEdge pose_edge,
    const v2::BrownConradyDistortionModel<double>& camera) -> void
{
  auto& logger = Logger::get();

  SARA_LOGD(logger, "Transform feature tracks into best feature pairs...");
  const auto pose_u = _pose_graph.source(pose_edge);
  const auto pose_v = _pose_graph.target(pose_edge);
  const auto& tsfm_u = _pose_graph[pose_u].pose;
  const auto& tsfm_v = _pose_graph[pose_v].pose;
  SARA_LOGD(logger, "Pose[{}]:\n{}", pose_u, tsfm_u.matrix34());
  SARA_LOGD(logger, "Pose[{}]:\n{}", pose_v, tsfm_v.matrix34());

  const auto num_tracks = static_cast<Eigen::Index>(tracks.size());

  using FeatureVertexPair = std::array<FeatureVertex, 2>;
  auto matches = std::vector<FeatureVertexPair>(num_tracks);
  std::transform(
      tracks.begin(), tracks.end(), matches.begin(),
      [this, pose_u, pose_v](const FeatureTrack& track) -> FeatureVertexPair {
        // Non-maximum suppression.
        //
        // We do need to filter the feature tracks by non-maximum suppression.
        //
        // Even in the case where the pose graph contains only 2 views, feature
        // matches can be merged into components of cardinality larger than 2.
        const auto track_filtered = filter_by_non_max_suppression(track);
        if (track_filtered.size() != 2)
          throw std::runtime_error{
              "Error: the NMS-filtered feature track must have cardinality 2!"};

        // Retrieve the cleaned up feature correspondence.
        const auto fu = find_feature_vertex_at_pose(track_filtered, pose_u);
        const auto fv = find_feature_vertex_at_pose(track_filtered, pose_v);
        if (!fu.has_value() || !fv.has_value())
          throw std::runtime_error{
              "Error: the feature match must exist in the graph!"};

        return {*fu, *fv};
      });

  SARA_LOGD(logger, "Calculating ray pairs from feature pairs...");
  auto rays_u = Eigen::MatrixXd{3, num_tracks};
  auto rays_v = Eigen::MatrixXd{3, num_tracks};
  for (auto t = 0u; t < num_tracks; ++t)
  {
    // Collect the feature match '(x, y)'.
    const auto& [x, y] = matches[t];
    const auto x_coords = pixel_coords(x).cast<double>();
    const auto y_coords = pixel_coords(y).cast<double>();

    // Backproject the pixel coordinates to their corresponding incident rays on
    // the camera plane.
    rays_u.col(t) = camera.backproject(x_coords);
    rays_v.col(t) = camera.backproject(y_coords);
  }

  // Calculate the associated triangulation.
  SARA_LOGD(logger, "Initialization the point cloud by 3D triangulation from "
                    "the relative pose...");
  const auto motion = normalized_camera(_pose_graph[pose_edge].motion).matrix();
  if ((tsfm_v.matrix34() - motion).norm() > 1e-6)
    throw std::runtime_error{"Error: the target abs pose must be initialized "
                             "as the relative motion!"};

  const auto [X, scales_u, scales_v] = triangulate_linear_eigen(  //
      tsfm_u.matrix34(), tsfm_v.matrix34(),                       //
      rays_u, rays_v);

  // Allocate the mapping from the feature vertices to the scene point index.
  if (!_from_vertex_to_scene_point_index.empty())
    _from_vertex_to_scene_point_index.clear();

  // Calculate the initial point cloud.
  if (!_point_cloud.empty())
    _point_cloud.clear();

  auto scene_point_index = ScenePointIndex{};
  for (auto j = 0; j < X.cols(); ++j)
  {
    // Only consider **cheiral** inliers:
    //
    // The triangulated 3D points must be in front of the two cameras!
    if (!(scales_u(j) > 0 && scales_v(j) > 0))
      continue;

    // Calculate the scene point.
    const Eigen::Vector3d coords = X.col(j).hnormalized();
    const auto color = retrieve_scene_point_color(coords, image,  //
                                                  tsfm_v, camera);

    // Store the scene point to the point cloud.
    auto scene_point_value = ScenePoint::Value{};
    scene_point_value << coords, color;
    _point_cloud.emplace_back(scene_point_value);

    // Recall that a match is a pair of feature vertex.
    const auto& [x, y] = matches[j];

    // Assign a scene point index to the two feature vertices.
    _from_vertex_to_scene_point_index[x] = scene_point_index;
    _from_vertex_to_scene_point_index[y] = scene_point_index;
    ++scene_point_index;
  }

  SARA_LOGD(logger, "point cloud: {} 3D points", _point_cloud.size());
}

auto PointCloudGenerator::propagate_scene_point_indices(
    const std::vector<FeatureTrack>& tracks) -> void
{
  auto& logger = Logger::get();

  SARA_LOGI(logger,
            "Propagating scene point indices to new feature vertices...");

  for (const auto& track : tracks)
  {
    const auto scene_point_indices = list_scene_point_indices(track);
    if (scene_point_indices.empty())
      continue;

#if defined(DEBUG_ME)
    if (scene_point_indices.size() > 1)
    {
      SARA_LOGT(logger, "Found a fused feature track...");

      using ScenePointIndexVector = Eigen::RowVector<  //
          ScenePointIndex, Eigen::Dynamic>;
      using FeatureVerticesAsVector = Eigen::Map<
          const Eigen::RowVector<FeatureVertexIndex, Eigen::Dynamic>>;
      using ScenePointIndicesAsVector = Eigen::Map<  //
          const Eigen::RowVector<ScenePointIndex, Eigen::Dynamic>>;

      const ScenePointIndexVector track_vector =
          FeatureVerticesAsVector(track.data(), track.size())
              .cast<ScenePointIndex>();

      const ScenePointIndexVector scene_index_vector =
          ScenePointIndicesAsVector(scene_point_indices.data(),
                                    scene_point_indices.size());
      SARA_LOGT(logger, "track indices: {}", track_vector);
      SARA_LOGT(logger, "scene point indices: {}", scene_index_vector);

      for (const auto& i : scene_point_indices)
        SARA_LOGT(logger, "scene coords[{}]: {}", i,
                  Eigen::RowVector3d(_point_cloud[i].coords().transpose()));
    }
#endif

    // 1. Calculating the barycentric scene point coordinates to disambiguate
    // the cluster of scene points.
    const auto scene_point = barycenter(scene_point_indices);
    for (const auto& i : scene_point_indices)
      _point_cloud[i] = scene_point;

    // 2. Assigning a unique scene point index for each vertex of the feature
    // track.
    const auto& scene_point_index = scene_point_indices.front();
    for (const auto& v : track)
      _from_vertex_to_scene_point_index[v] = scene_point_index;
  }
}

auto PointCloudOperator::compress_point_cloud(
    const std::vector<FeatureTrack>& tracks) -> bool
{
  auto& logger = Logger::get();
  SARA_LOGI(logger, "Compressing the point cloud...");

  // Calculate the barycentric scene point for a given feature track.
  auto point_cloud_compressed = std::vector<ScenePoint>{};
  point_cloud_compressed.reserve(tracks.size());

  // Reset the scene point index for each feature track.
  for (auto t = ScenePointIndex{}; t < tracks.size(); ++t)
  {
    const auto scene_point_indices = list_scene_point_indices(tracks[t]);
    if (scene_point_indices.empty())
      continue;

    // Reassign the scene point index for the given feature track.
    for (const auto& v : track)
      _from_vertex_to_scene_point_index[v] = t;

    // Recalculate the scene point index as a barycenter.
    const auto scene_point =
        barycenter(scene_point_indices)
            point_cloud_compressed.emplace_back(scene_point);
  }

  // Swap the point cloud with the set of barycenters.
  std::swap(_point_cloud, point_cloud_compressed);

  return true;
}
