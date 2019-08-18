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

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/incremental_components.hpp>
#include <boost/pending/disjoint_sets.hpp>


namespace DO::Sara {

// This is a necessary step for the bundle adjustment step.

//! @brief Feature GID.
struct PoseID
{
  double weight{0};
};

struct EdgeID
{
  int id{-1};
  double weight{0};
};

using PoseGraph = boost::adjacency_list<boost::vecS, boost::vecS,
                                        boost::undirectedS, PoseID, EdgeID>;

} /* namespace DO::Sara */
