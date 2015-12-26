// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/DisjointSets/DisjointSets.hpp>


using namespace std;


namespace DO { namespace Sara {

  void
  DisjointSets::compute_connected_components()
  {
    for (vertex_type v = 0; v != _vertices.size(); ++v)
      make_set(v);

    for (vertex_type u = 0; u != _vertices.size(); ++u)
    {
      adjacent_vertex_iterator v, v_end;
      v     = _adj_list.out_vertices_begin(u);
      v_end = _adj_list.out_vertices_end(u);

      for ( ; v != v_end; ++v)
        join(&_vertices[u], &_vertices[*v]);
    }
  }

  std::vector<std::vector<DisjointSets::vertex_type>>
  DisjointSets::get_connected_components() const
  {
    unordered_map<vertex_type, vector<vertex_type>> table;
    for (size_t v = 0; v != _vertices.size(); ++v)
    {
      size_t c = component(v);
      table[c].push_back(v);
    }

    vector<vector<vertex_type>> components;
    components.reserve(table.size());
    for (const auto& c : table)
      components.push_back(c.second);
    return components;
  }

} /* namespace Sara */
} /* namespace DO */
