#pragma once

#include <DO/Sara/DisjointSets.hpp>

#include <vector>

namespace DO::Sara {

  // This is a necessary step for the bundle adjustment step.

  struct GlobalID {
    int image_idx;
    int feature_idx;
  };

  struct PointTrackGraph
  {
    //auto init(std::vector<KeypointList<OERegion, float>>& keys,
    //          EpipolarGraph& g)
    //{
    //}

    auto flat_id(const GlobalID& gid) const
    {
      return gid.image_idx * max_num_points_per_image + gid.feature_idx;
    }

    auto calculate_num_points_per_image()
    {
      if (num_features.empty())
        max_num_points_per_image = 0;
      else
        max_num_points_per_image =
            *std::max_element(std::begin(num_features), std::end(num_features));
    }

    int num_images;
    std::vector<int> num_features;

    int max_num_points_per_image;

    std::vector<GlobalID> gids;
    std::vector<bool> exists;
    AdjacencyList adj_list;
  };


} /* namespace DO::Sara */
