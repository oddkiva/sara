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

#include <DO/Sara/MultiViewGeometry/Camera/v2/PinholeCamera.hpp>
#include <DO/Sara/SfM/BuildingBlocks/RgbColoredPoint.hpp>
#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>
#include <DO/Sara/SfM/Graph/FeatureTracker.hpp>

#include <filesystem>


namespace DO::Sara {

  class PointCloudGenerator
  {
  public: /* aliases */
    using PoseVertex = CameraPoseGraph::Vertex;
    using PoseEdge = CameraPoseGraph::Edge;

    using FeatureVertex = FeatureGraph::Vertex;
    using FeatureVertexIndex = FeatureGraph::VertexIndex;

    using ScenePointIndex = std::size_t;
    using ScenePoint = RgbColoredPoint<double>;

    using PointCloud = std::vector<RgbColoredPoint<double>>;
    using FeatureTrack = FeatureTracker::Track;
    using FeatureToScenePointMap = std::unordered_map<FeatureVertex,  //
                                                      ScenePointIndex>;

  public: /* constructor */
    PointCloudGenerator(const CameraPoseGraph& camera_pose_graph,
                        const FeatureGraph& feature_graph,
                        PointCloud& point_cloud)
      : _pose_graph{camera_pose_graph}
      , _feature_graph{feature_graph}
      , _point_cloud{point_cloud}
    {
    }

  public: /* helper feature retrieval methods */
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

  public: /* helper query methods */
    auto list_scene_point_indices(const FeatureTrack&) const
        -> std::vector<ScenePointIndex>;

    auto find_feature_vertex_at_pose(const FeatureTrack&,  //
                                     const PoseVertex) const
        -> std::optional<FeatureVertex>;

    auto
    retrieve_scene_point_color(const Eigen::Vector3d& scene_point,  //
                               const ImageView<Rgb8>& image,        //
                               const QuaternionBasedPose<double>& pose,
                               const v2::PinholeCamera<double>& camera) const
        -> Rgb64f;

  public: /* data transformation methods */
    //! @brief Calculate the barycentric scene point.
    //!
    //! We expect the array of scene point indices to be originated from a
    //! feature track.
    auto barycenter(const std::vector<ScenePointIndex>&) const -> ScenePoint;

    //! @brief A track is a sequence of feature.
    auto filter_by_non_max_suppression(const FeatureTrack&) const  //
        -> FeatureTrack;

    //! @brief Split the list of feature tracks into two lists.
    //!
    //! The first list contains the tracks for which a scene point is
    //! calculated. The second list contains the tracks for which a scene point
    //! is not yet calculated.
    auto split_by_scene_point_knowledge(const std::vector<FeatureTrack>&) const
        -> std::pair<std::vector<FeatureTrack>, std::vector<FeatureTrack>>;

    //! - The point cloud compression reassigns a unique scene point cloud to
    //!   each feature tracks.
    //! - The scene point is recalculated as a the barycenter of the
    //!   possibly multiple scene points we have found after recalculating the
    //!   feature tracks.
    auto propagate_scene_point_indices(const std::vector<FeatureTrack>&)
        -> void;

    //! - The point cloud compression reassigns a unique scene point cloud to
    //!   each feature tracks.
    //! - The scene point is recalculated as a the barycenter of the
    //!   possibly multiple scene points we have found after recalculating the
    //!   feature tracks.
    auto compress_point_cloud(const std::vector<FeatureTrack>&) -> bool;

    //! Grow the point cloud becomes possible when the most recent absolute pose
    //! is known.
    //!
    //! This calculates the new 3D scene points. Specifically the new 3D scene
    //! points are those calculated from the feature tracks for which we didn't
    //! know their scene point values.
    auto grow_point_cloud(
        const std::vector<FeatureTrack>& ftracks_without_scene_point,
        const ImageView<Rgb8>& image,  //
        const PoseEdge pose_edge,      //
        const v2::PinholeCamera<double>& camera) -> void;

    auto save_point_cloud(const std::filesystem::path& out_csv_path) const
        -> void;

  private: /* data members */
    const CameraPoseGraph& _pose_graph;
    const FeatureGraph& _feature_graph;
    PointCloud& _point_cloud;

    FeatureToScenePointMap _from_vertex_to_scene_point_index;
  };

}  // namespace DO::Sara
