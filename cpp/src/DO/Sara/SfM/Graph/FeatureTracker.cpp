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
    const CameraPoseGraph& camera_pose_graph,
    const CameraPoseGraph::Edge relative_pose_edge) -> void
{
  auto& logger = Logger::get();

  const CameraPoseGraph::Impl& cg = camera_pose_graph;
  FeatureGraph::Impl& fg = _feature_graph;

  // Retrieve the two camera vertices from the relative pose edge.
  const auto pose_i = boost::source(relative_pose_edge, cg);
  const auto pose_j = boost::target(relative_pose_edge, cg);
  // The relative pose edge contains the set of all feature correspondences.
  const auto& matches = cg[relative_pose_edge].matches;
  // Which of these feature correspondences are marked as inliers?
  const auto& inliers = cg[relative_pose_edge].inliers;

  // Loop over the feature correspondence and add the feature graph edges.
  SARA_LOGD(logger, "Pose {} <-> Pose {}", pose_i, pose_j);
  SARA_LOGD(logger, "Add feature correspondences...");
  for (auto m = 0u; m < matches.size(); ++m)
  {
    if (!inliers(m))
      continue;

    const auto& match = matches[m];

    // Local feature indices.
    const auto& f1 = match.x_index();
    const auto& f2 = match.y_index();

    // Create their corresponding feature GIDs.
    const auto gid1 = FeatureGID{
        .pose_vertex = pose_i,  //
        .feature_index = f1     //
    };
    const auto gid2 = FeatureGID{
        .pose_vertex = pose_j,  //
        .feature_index = f2     //
    };

    // Locate their corresponding pair of vertices (u, v) in the graph?
    // Do they exist yet in the first place?
    const auto u_it = _feature_vertex.find(gid1);
    const auto v_it = _feature_vertex.find(gid2);

    const auto u_does_not_exist_yet = u_it == _feature_vertex.end();
    const auto v_does_not_exist_yet = v_it == _feature_vertex.end();

    // If not, add them if necessary.
    const auto u = u_does_not_exist_yet ? boost::add_vertex(fg) : u_it->second;
    const auto v = v_does_not_exist_yet ? boost::add_vertex(fg) : v_it->second;

    if (u_does_not_exist_yet)
    {
      fg[u] = gid1;
      _feature_vertex[gid1] = u;
    }
    if (v_does_not_exist_yet)
    {
      fg[v] = gid2;
      _feature_vertex[gid2] = v;
    }

    // Finally, store the feature match as an edge in the feature graph.
    const auto [uv, uv_added] = boost::add_edge(u, v, fg);
    auto& uv_attrs = fg[uv];
    uv_attrs.i = boost::source(relative_pose_edge, cg);
    uv_attrs.j = boost::target(relative_pose_edge, cg);
    uv_attrs.index = m;
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
