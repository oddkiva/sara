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

#pragma once

#include <vector>

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Image.hpp>


namespace DO { namespace Sara {

  //! @brief Adjacency list.
  class AdjacencyList
  {
  public:
    using size_type = std::size_t;
    using vertex_type = std::size_t;
    using out_vertex_iterator = std::vector<size_type>::iterator;
    using const_out_vertex_iterator = std::vector<size_type>::const_iterator;

    AdjacencyList(std::vector<std::vector<vertex_type>>& adjacency_data)
      : _a(adjacency_data)
    {
    }

    size_type out_degree(vertex_type v) const
    {
      return _a[v].size();
    }

    out_vertex_iterator out_vertices_begin(vertex_type v)
    {
      return _a[v].begin();
    }

    out_vertex_iterator out_vertices_end(vertex_type v)
    {
      return _a[v].end();
    }

    const_out_vertex_iterator out_vertices_begin(vertex_type v) const
    {
      return _a[v].begin();
    }

    const_out_vertex_iterator out_vertices_end(vertex_type v) const
    {
      return _a[v].end();
    }

  private:
    std::vector<std::vector<size_type> >& _a;
  };


  DO_SARA_EXPORT
  std::vector<std::vector<std::size_t>>
  compute_adjacency_list_2d(const ImageView<int>& labels);


} /* namespace Sara */
} /* namespace DO */
