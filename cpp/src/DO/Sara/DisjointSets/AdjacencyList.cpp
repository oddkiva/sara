// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/DisjointSets/AdjacencyList.hpp>


using namespace std;


namespace DO { namespace Sara {

  vector<vector<size_t>> compute_adjacency_list_2d(const ImageView<int>& labels)
  {
    const auto size = labels.size();
    const int dx[4] = { 1, 0,-1, 0 };
    const int dy[4] = { 0, 1, 0,-1 };

    auto A = vector<vector<size_t>>(size);
    for (auto& row_i : A)
      row_i.reserve(4);

    for (int y = 0; y < labels.height(); ++y)
    {
      for (int x = 0; x < labels.width(); ++x)
      {
        const auto p = x + y * labels.width();
        const auto l_p = labels(x, y);

        for (int i = 0; i < 4; ++i)
        {
          const auto x_q = x + dx[i];
          if (x_q < 0 || x_q == labels.width())
            continue;

          const auto y_q = y + dy[i];
          if (y_q < 0 || y_q == labels.height())
            continue;

          const auto q = x_q + y_q * labels.width();
          const auto l_q = labels(x_q, y_q);

          if (l_p == l_q)
            A[p].push_back(q);
        }
      }
    }

    return A;
  }

} /* namespace Sara */
} /* namespace DO */
