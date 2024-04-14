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

#include <DO/Sara/SfM/Graph/FeatureDisjointSets.hpp>

#include <map>


namespace DO::Sara {

  //! @brief Feature tracker
  struct FeatureTracker
  {
    using Track = std::vector<FeatureGraph::VertexIndex>;
    using TrackArray = std::vector<Track>;
    using TrackVisibilityCountArray = std::vector<std::size_t>;

    //! @brief The graph of 2D image features
    //!
    //! Image features are connected if there exists a relative pose that
    //! explains it.
    FeatureGraph _feature_graph;
    FeatureDisjointSets _feature_ds;
    std::vector<Track> _feature_tracks;

    //! @brief Retrieve the feature vertex from its pair (camera pose vertex,
    //! keypoint index).
    std::map<FeatureGID, FeatureGraph::Vertex> _feature_vertex;

    //! @brief Retrieve the feature edge from its pair (camera pose edge,
    //! match index).
    std::map<MatchGID, FeatureGraph::Edge> _feature_match;

    auto update_feature_tracks(const CameraPoseGraph&,
                               const CameraPoseGraph::Edge) -> void;

    auto calculate_alive_feature_tracks(
        const CameraPoseGraph::Vertex last_pose_vertex) const
        -> std::tuple<TrackArray, TrackVisibilityCountArray>;
  };

}  // namespace DO::Sara
