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

#pragma once

#include <DO/Sara/SfM/BuildingBlocks/RgbColoredPoint.hpp>
#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>
#include <DO/Sara/SfM/Graph/FeatureTracker.hpp>


namespace DO::Sara {

  class PointCloudGenerator
  {
  public:
    using PoseVertex = CameraPoseGraph::Vertex;
    using PoseEdge = CameraPoseGraph::Edge;
    using FeatureVertex = FeatureGraph::Vertex;
    using ScenePointIndex = std::size_t;
    using ScenePoint = RgbColoredPoint<double>;

    using PointCloud = std::vector<RgbColoredPoint<double>>;
    using FeatureTrack = FeatureTracker::Track;

    PointCloudGenerator(const CameraPoseGraph& camera_pose_graph,
                        const FeatureGraph& feature_graph,
                        PointCloud& point_cloud)
      : _pose_graph{camera_pose_graph}
      , _feature_graph{feature_graph}
      , _point_cloud{point_cloud}
    {
    }

    auto list_scene_point_indices(const FeatureTrack&) const
        -> std::vector<ScenePointIndex>;

    auto filter_by_non_max_suppression(const FeatureTrack&) const  //
        -> FeatureTrack;

    auto split_by_scene_point_knowledge(const std::vector<FeatureTrack>&) const
        -> std::pair<std::vector<FeatureTrack>, std::vector<FeatureTrack>>;

    auto seed_point_cloud_from_two_views(
        const std::vector<FeatureTrack>&,  //
        const ImageView<Rgb8>&,            //
        const PoseEdge,                    //
        const v2::BrownConradyDistortionModel<double>&) -> void;

  public: /* utility methods */
    auto gid(const FeatureVertex u) const -> const FeatureGID&
    {
      return _feature_graph[u];
    }

    auto feature(const FeatureVertex u) const -> const OERegion&
    {
      const auto& [pose_vertex, feature_index] = gid(u);
      const auto& f = features(_pose_graph[pose_vertex].keypoints);
      return f[feature_index];
    }

    auto pixel_coords(const FeatureVertex u) const -> const Eigen::Vector2f&
    {
      return feature(u).center();
    }

    auto barycenter(const std::vector<ScenePointIndex>&) const -> ScenePoint;

    auto find_feature_vertex_at_pose(const FeatureTrack&,  //
                                     const PoseVertex) const
        -> std::optional<FeatureVertex>;

    auto retrieve_scene_point_color(
        const ScenePoint::Coords& scene_point_coords,      //
        const ImageView<Rgb8>& image,                      //
        const QuaternionBasedPose<double>& absolute_pose,  //
        const v2::BrownConradyDistortionModel<double>& camera) const
        -> ScenePoint::Color;

  private:
    const CameraPoseGraph& _pose_graph;
    const FeatureGraph& _feature_graph;
    PointCloud& _point_cloud;

    std::unordered_map<FeatureVertex, ScenePointIndex>
        _from_vertex_to_scene_point_index;
  };

}  // namespace DO::Sara
