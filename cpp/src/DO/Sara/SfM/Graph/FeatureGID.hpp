// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>
#include <utility>


namespace DO::Sara {

  //! @brief Feature Global ID (GID).
  struct FeatureGID
  {
    CameraPoseGraph::Vertex pose_vertex;
    std::size_t feature_index;

    auto operator==(const FeatureGID& other) const -> bool
    {
      return pose_vertex == other.pose_vertex &&
             feature_index == other.feature_index;
    }

    auto operator<(const FeatureGID& other) const -> bool
    {
      return std::make_pair(pose_vertex, feature_index) <
             std::make_pair(other.pose_vertex, other.feature_index);
    }
  };

}  // namespace DO::Sara
