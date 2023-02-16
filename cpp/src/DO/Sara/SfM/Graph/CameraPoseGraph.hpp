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

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Core/Pixel.hpp>
#include <DO/Sara/SfM/Graph/ImageFeatures.hpp>

#include <boost/graph/adjacency_list.hpp>

#include <optional>


namespace DO::Sara {

  struct RelativePoseEdge
  {
    int src_camera_view_id = -1;
    int dst_camera_view_id = -1;

    std::vector<std::pair<int, int>> _matches;
    std::vector<std::uint8_t> _inliers;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
  };

  struct CameraPoseGraph
  {
    using GraphImpl =
        boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                              CameraPose<double>, RelativePoseEdge>;
    using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
    using Edge = boost::graph_traits<Graph>::edge_descriptor;

    GraphImpl _pose_graph;
    std::vector<std::optional<Image<Rgb8>>> _images;
    ImageFeatures _image_features;
  };

} /* namespace DO::Sara */
