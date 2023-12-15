#pragma once

#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>
#include <DO/Sara/SfM/Graph/FeatureGraph.hpp>
#include <DO/Sara/SfM/Graph/RgbColoredPoint.hpp>


namespace DO::Sara {

  class PointCloudManipulator
  {
  public:
    using scene_point_index_t = std::size_t;
    using PointCloud = std::vector<RgbColoredPoint<double>>;

    PointCloudManipulator(const CameraPoseGraph& camera_pose_graph,
                          const FeatureGraph& feature_graph,
                          PointCloud& point_cloud)
      : _camera_pose_graph{camera_pose_graph}
      , _feature_graph{feature_graph}
      , _point_cloud{point_cloud}
    {
    }

    auto grow_point_cloud(const std::vector<FeatureGraph::Track>&,
                          const CameraPoseGraph::Edge&,  //
                          const ImageView<Rgb8>&) -> void;

    auto compact_point_cloud(const std::vector<FeatureGraph::Track>&,
                             const CameraPoseGraph::Edge&,
                             const ImageView<Rgb8>&) -> void;

  private:
    const CameraPoseGraph& _camera_pose_graph;
    const FeatureGraph& _feature_graph;
    PointCloud& _point_cloud;

    std::unordered_map<FeatureGraph::Vertex, scene_point_index_t>
        _from_vertex_to_scene_point_index;
  };

}  // namespace DO::Sara
