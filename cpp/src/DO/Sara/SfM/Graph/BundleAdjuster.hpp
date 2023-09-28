#pragma once

#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>
#include <DO/Sara/SfM/Graph/FeatureGraph.hpp>
#include <DO/Sara/SfM/Graph/PointCloudManipulator.hpp>


namespace DO::Sara {

  class BundleAdjuster
  {
  public:
    BundleAdjuster(CameraPoseGraph& camera_pose_graph,
                   const FeatureGraph& feature_graph,
                   PointCloudManipulator::PointCloud& point_cloud)
      : _camera_pose_graph{camera_pose_graph}
      , _feature_graph{feature_graph}
      , _point_cloud{point_cloud}
    {
    }

    auto adjust() -> void;

  private:
    CameraPoseGraph& _camera_pose_graph;
    const FeatureGraph& _feature_graph;
    PointCloud& _point_cloud;
  };

}  // namespace DO::Sara
