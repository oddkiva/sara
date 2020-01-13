// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/HDF5.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/incremental_components.hpp>
#include <boost/pending/disjoint_sets.hpp>


namespace DO::Sara {

  //! @addtogroup MultiViewGeometry
  //! @{

  //! @{
  //! @brief Pose graph data structures.
  struct PoseID
  {
    double weight{0};
  };

  struct EpipolarEdgeID
  {
    int id{-1};
    double weight{0};
  };

  using PoseGraph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                            PoseID, EpipolarEdgeID>;
  //! @}


  //! @brief write feature graph to HDF5.
  DO_SARA_EXPORT
  auto write_pose_graph(const PoseGraph& graph, H5File& file,
                        const std::string& group_name) -> void;

  //! @brief read feature graph from HDF5.
  DO_SARA_EXPORT
  auto read_pose_graph(H5File& file, const std::string& group_name)
      -> PoseGraph;

  //! @}

} /* namespace DO::Sara */
