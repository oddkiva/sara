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

#include <DO/Sara/SfM/Graph/FeatureTracker.hpp>

#include <DO/Sara/Logging/Logger.hpp>

#include <boost/foreach.hpp>


using namespace DO::Sara;

auto FeatureTracker::update_feature_tracks(
    const CameraPoseGraph& pose_graph, const CameraPoseGraph::Edge pose_edge)
    -> void
{
  auto& logger = Logger::get();

  // Retrieve the camera poses from the relative pose edge.
  const auto pose_u = pose_graph.source(pose_edge);
  const auto pose_v = pose_graph.target(pose_edge);
  // The relative pose edge contains the set of all feature correspondences.
  const auto& matches = pose_graph[pose_edge].matches;
  // Which of these feature correspondences are marked as inliers?
  const auto& inliers = pose_graph[pose_edge].inliers;

  // Add the feature graph edges.
  //
  // They are feature matches that are deemed inliers according the relative
  // pose estimation task.
  SARA_LOGD(logger, "Pose {} <-> Pose {}", pose_u, pose_v);
  SARA_LOGD(logger, "Add feature correspondences...");
  for (auto m = 0u; m < matches.size(); ++m)
  {
    if (!inliers(m))
      continue;

    // The feature match is 'm = (ix, iy)'
    // where 'ix' and 'iy' are the local IDs of feature 'x' and 'y'.
    const auto& match = matches[m];

    // 'x' and 'y' are respectively identified by their GID 'gid_x' and 'gid_y',
    // which are defined as follows.
    const auto gid_x = FeatureGID{
        .pose_vertex = pose_u,            //
        .feature_index = match.x_index()  //
    };
    const auto gid_y = FeatureGID{
        .pose_vertex = pose_v,            //
        .feature_index = match.y_index()  //
    };

    // Are features 'x' and 'y' already added in the graph, i.e.,
    // are vertex 'gid_x' and 'gid_y' already added in the graph?
    const auto it_x = _feature_vertex.find(gid_x);
    const auto it_y = _feature_vertex.find(gid_y);

    const auto x_does_not_exist_yet = it_x == _feature_vertex.end();
    const auto y_does_not_exist_yet = it_y == _feature_vertex.end();

    // If not, add them if necessary.
    FeatureGraph::Impl& fg = _feature_graph;
    const auto x = x_does_not_exist_yet ? boost::add_vertex(fg) : it_x->second;
    const auto y = y_does_not_exist_yet ? boost::add_vertex(fg) : it_y->second;

    if (x_does_not_exist_yet)
    {
      fg[x] = gid_x;
      _feature_vertex[gid_x] = x;
    }
    if (y_does_not_exist_yet)
    {
      fg[y] = gid_y;
      _feature_vertex[gid_y] = y;
    }

    // Finally, store the feature match as an edge in the feature graph to
    // navigate between the feature graph to the pose graph.
    const auto [xy, xy_added] = boost::add_edge(x, y, fg);
    auto& xy_attrs = fg[xy];
    xy_attrs.pose_src = pose_graph.source(pose_edge);
    xy_attrs.pose_dst = pose_graph.target(pose_edge);
    xy_attrs.index = m;
  }

  // Update the feature disjoint-sets
  SARA_LOGD(logger, "[Feature-Tracks] Recalculating connected components...");
  const auto _feature_ds = FeatureDisjointSets{_feature_graph};
  const auto feature_components = _feature_ds.components();
  SARA_LOGD(logger, "[Feature-Tracks] num feature components = {}",
            feature_components.size());

  // Update the list of feature tracks.
  _feature_tracks.clear();
  _feature_tracks.reserve(feature_components.size());

  BOOST_FOREACH (FeatureGraph::VertexIndex current_index, feature_components)
  {
    // Iterate through the child vertex indices for [current_index]
    auto component_size = 0;
    BOOST_FOREACH (FeatureGraph::VertexIndex child_index,
                   feature_components[current_index])
    {
      (void) child_index;
      ++component_size;
    }

    if (component_size == 1)
      continue;

    auto track = std::vector<FeatureGraph::VertexIndex>{};
    track.reserve(component_size);
    BOOST_FOREACH (FeatureGraph::VertexIndex child_index,
                   feature_components[current_index])
      track.push_back(child_index);

    _feature_tracks.emplace_back(std::move(track));
  }

  SARA_LOGD(logger, "[Feature-Tracks] num feature tracks = {}",
            _feature_tracks.size());
}

auto FeatureTracker::calculate_alive_feature_tracks(
    const CameraPoseGraph::Vertex camera_vertex_curr) const
    -> std::tuple<TrackArray, TrackVisibilityCountArray>
{
  auto& logger = Logger::get();

  // Find the feature tracks that are still alive.
  const FeatureGraph::Impl& fgraph = _feature_graph;

  const auto& ftracks = _feature_tracks;
  auto tracks_alive = TrackArray{};
  auto track_visibility_count = TrackVisibilityCountArray{};

  for (const auto& ftrack : ftracks)
  {
    // Do we still see the track in the image.
    const auto is_alive =
        std::find_if(ftrack.begin(), ftrack.end(),
                     [&fgraph, camera_vertex_curr](const auto& v) {
                       return fgraph[v].pose_vertex == camera_vertex_curr;
                     }) != ftrack.end();

    if (!is_alive)
      continue;

    // Add the newly found alive track.
    tracks_alive.push_back(ftrack);

    // Carefully count the track life, it's not the number of vertices, but
    // the number of camera views in which the feature reappears.
    auto camera_vertices_where_present =
        std::unordered_set<CameraPoseGraph::Vertex>{};
    std::transform(ftrack.begin(), ftrack.end(),
                   std::inserter(camera_vertices_where_present,
                                 camera_vertices_where_present.end()),
                   [&fgraph](const auto& v) { return fgraph[v].pose_vertex; });
    track_visibility_count.push_back(camera_vertices_where_present.size());
  }
  SARA_LOGD(logger, "Num tracks alive: {}", tracks_alive.size());

  const auto longest_track_alive = std::max_element(
      track_visibility_count.begin(), track_visibility_count.end());
  if (longest_track_alive != track_visibility_count.end())
  {
    SARA_LOGD(logger, "Longest track life: {}", *longest_track_alive);
#if 0
       const auto longest_track_index =
           longest_track_alive - track_visibility_count.begin();
       const auto& longest_track = tracks_alive[longest_track_index];
       for (const auto& v : longest_track)
         std::cout << fmt::format("(cam:{},ind:{})", fgraph[v].camera_vertex,
                                  fgraph[v].index)
                   << " ";
       std::cout << std::endl;
#endif
  }

  return std::make_tuple(tracks_alive, track_visibility_count);
}
