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
#include <DO/Sara/MultiViewGeometry/Geometry/QuaternionBasedPose.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>
#include <DO/Sara/SfM/Graph/ImageFeatures.hpp>

#include <boost/graph/adjacency_list.hpp>

#include <optional>
#include <vector>


namespace DO::Sara {

  struct RelativePoseEdge
  {
    using camera_id_t = int;
    static constexpr auto undefined_camera_id = -1;

    camera_id_t src_camera = undefined_camera_id;
    camera_id_t dst_camera = undefined_camera_id;

    std::vector<std::pair<int, int>> _matches;
    std::vector<std::uint8_t> _inliers;
    Motion _motion;
  };

  struct CameraPoseGraph
  {
    using GraphImpl =
        boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                              QuaternionBasedPose<double>, RelativePoseEdge>;
    using Vertex = boost::graph_traits<GraphImpl>::vertex_descriptor;
    using Edge = boost::graph_traits<GraphImpl>::edge_descriptor;

    GraphImpl _pose_graph;
    std::vector<std::optional<Image<Rgb8>>> _images;
    ImageKeypoints _image_keypoints;
  };

} /* namespace DO::Sara */
