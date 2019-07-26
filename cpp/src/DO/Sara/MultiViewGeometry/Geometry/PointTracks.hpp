#pragma once

#include <boost/graph/connected_components.hpp>

#include <vector>


namespace DO::Sara {

  // This is a necessary step for the bundle adjustment step.

  //! @brief Point Global ID.
  struct PointGID {
    int image_idx{-1};
    int local_idx{-1};
  };

  struct PointTrackGraph
  {
    auto flat_id(const PointGID& gid) const
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
  };


} /* namespace DO::Sara */
