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

//! @file

#include <DO/Sara/Core/Image.hpp>

#include <numeric>


namespace DO::Sara {


  namespace dijkstra_details {

    template <typename Graph>
    auto initialize(Graph& g)
    {
      auto v = vertices(g);
      auto d = distances(V);
      auto p = predecessor(V);
      for (auto& v : v)
      {
        d[v] = std::numeric_limits<double>::max();
        p[v] = nullptr;
      }
    }

    template <typename Vertex, typename Graph, typename Weight>
    auto relax(Vertex u, Vertex v, Graph& g, Weight& w)
    {
      auto V = vertices(G);

      auto d = distances(V);
      auto p = predecessor(V);

      const auto tentative_dv = d[u] + w(u, v);
      if (d[v] > tentative_dv)
      {
        d[v] = tentative_dv;
        p[v] = u;
      }
    }

  }  // namespace dijkstra_details

  template <typename Graph, typename Weight>
  auto disjktra(Graph& g, Weight w)
  {
    using vertex_type = typename Graph::vertex_type;
    dijkstra_details::initialize(g);
    //dijkstra_details::set_sources(g);

    auto s = std::deque<vertex_type>{};
    auto q = std::priority_queue<vertex_type>{vertices(g)};

    while (!q.empty())
    {
      auto u = q.top();
      q.pop();
      s.push_back(u);

      for (auto& v : g[u])
      {
        dijkstra_details::relax(u, v, g, w);
      }
    }
  }


  enum class State : std::uint8_t
  {
    Alive = 0,
    Trial = 1,
    Far = 2,
    Forbidden = 3
  };


  template <typename T, int N>
  struct ImageBasedGraph
  {
    using coords_type = Matrix<int, N, 1>;

    struct CoordsVal
    {
      T val;
      coords_type coords;

      inline bool operator<(CoordsVal& other) const
      {
        return val < other.val;
      }
    };

    Image<State, N> states;
    Image<T, N> distances;
    Image<std::optional<coords_type>, N> predecessors;
  };


  template <typename T, int N, typename WeightFunction>
  struct FastMarching
  {
    using value_type = T;
    using image_based_graph_type = ImageBasedGraph<T, N>;
    using coords_type = typename image_based_graph_type::coords_type;
    using coords_val_type = typename ImageBasedGraph<T, N>::CoordsVal;

    using trial_priority_queue_type = std::priority_queue<coords_val_type>;
    using alive_container_type = std::priority_queue<coords_val_type>;

    void initialize()
    {
      // Initialize all vertices.
      g.states.fill(State::Far);
      g.distances.fill(std::numeric_limits<T>::max());
      g.predecessors.fill({});

      trial = trial_priority_queue_type{};
      alive = alive_container_type{};
    }

    void relax(const coords_type& u, coords_type& v,
               image_based_graph_type& g)
    {
      const auto dv_candidate = g.distances[u] + w(u, v);
      if (g.distances[v] < dv_candidate)
      {
        g.distances[v] = dv_candidate;
        g.predecessors[v] = u;
        
      }
    }

    auto extract_min()
    {
      auto u = trial.top();
      std::pop_heap(std::begin(trial), std::end(trial));
      trial.pop_back();
      return u;
    }


    void operator()() const
    {
      initialize();

      // Initialize with trial points.
      //for (auto v : trial_list)
      //  trial.push_back(...);
      //std::make_heap(std::begin(trial), std::end(trial));

      while (!trial.empty())
      {
        // extract min.
        auto u = extract_min();

        // Update its state.
        alive.push_back(u);
        g.states[u.coords] = State::Alive;

        // @TODO: check if we reach the limit.


        // Update the neighbors.
        for (auto& v: g.neighbors(u))
        {
          if (g.states[v] == State::Far)
          {
            // Update state and value.
            relax(u.coords, v.coords, g);

            // Push in the heap.
            trial.push_back(v);
            std::push_heap(std::begin(trial), std::end(trial));
          }

          if (g.states[v] == State::Trial)
          {
            // Update state and value.
            relax(u.coords, v.coords, g);

            // Increase the point priority in the heap.
          }
        }

      }
    }

    graph_type g;

    // Weight function typically the Eikonal equation.
    WeightFunction w;

    // Dijkstra algorithm.
    std::vector<coords_val_type> trial;
    std::vector<coords_val_type> alive;
    T limit;
  };

}  // namespace DO::Sara
