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

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Features/Feature.hpp>
#include <DO/Sara/Features/KeypointList.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/QuaternionBasedPose.hpp>

#include <DO/Sara/SfM/BuildingBlocks/FeatureParams.hpp>
#include <DO/Sara/SfM/BuildingBlocks/RelativePoseEstimator.hpp>

#include <boost/graph/adjacency_list.hpp>


namespace DO::Sara {

  struct AbsolutePoseData
  {
    //! @brief The corresponding image frame index or image ID.
    int image_id;
    //! @brief The keypoints detected in the image.
    KeypointList<OERegion, float> keypoints;
    //! @brief "Absolute" pose w.r.t. some reference frame.
    QuaternionBasedPose<double> pose;

    AbsolutePoseData() = default;

    AbsolutePoseData(const AbsolutePoseData&) = default;

    AbsolutePoseData(AbsolutePoseData&&) = default;

    AbsolutePoseData(const int image_id,
                     KeypointList<OERegion, float>&& keypoints,
                     QuaternionBasedPose<double>&& pose)
      : image_id{image_id}
      , keypoints{std::move(keypoints)}
      , pose{std::move(pose)}
    {
    }

    auto operator=(AbsolutePoseData&&) -> AbsolutePoseData& = default;
  };

  struct RelativePoseData
  {
    std::vector<Match> matches;
    Tensor_<bool, 1> inliers;
    Motion motion;
  };

  class CameraPoseGraph
  {
  public:
    using Impl = boost::adjacency_list<                //
        boost::vecS, boost::vecS, boost::undirectedS,  //
        AbsolutePoseData, RelativePoseData>;
    using Vertex = boost::graph_traits<Impl>::vertex_descriptor;
    using VertexIndex = boost::graph_traits<Impl>::vertices_size_type;
    using Edge = boost::graph_traits<Impl>::edge_descriptor;

    operator Impl&()
    {
      return _g;
    }

    operator const Impl&() const
    {
      return _g;
    }

    auto operator[](const Vertex u) -> AbsolutePoseData&
    {
      return _g[u];
    }

    auto operator[](const Vertex u) const -> const AbsolutePoseData&
    {
      return _g[u];
    }

    auto operator[](const Edge e) -> RelativePoseData&
    {
      return _g[e];
    }

    auto operator[](const Edge e) const -> const RelativePoseData&
    {
      return _g[e];
    }

    auto source(const Edge e) const -> Vertex
    {
      return boost::source(e, _g);
    }

    auto target(const Edge e) const -> Vertex
    {
      return boost::target(e, _g);
    }


    auto edge(const Vertex u, const Vertex v) const -> std::pair<Edge, bool>
    {
      return boost::edge(u, v, _g);
    }

    auto num_vertices() const -> VertexIndex
    {
      return boost::num_vertices(_g);
    }

    auto add_absolute_pose(AbsolutePoseData&& data) -> Vertex;

    auto add_relative_pose(const Vertex src, const Vertex dst,
                           RelativePoseData&& relative_pose_data) -> Edge;

  private:
    //! @brief The graph data structure shortened as g.
    Impl _g;
  };

} /* namespace DO::Sara */
