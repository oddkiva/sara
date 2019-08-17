#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>


namespace DO::Sara {

// This is a necessary step for the bundle adjustment step.

//! @brief Feature GID.
struct FeatureGID
{
  int image_id{-1};
  int local_id{-1};
};

using FeatureGraph = boost::adjacency_list<boost::vecS, boost::vecS,
                                           boost::undirectedS, FeatureGID>;


} /* namespace DO::Sara */
