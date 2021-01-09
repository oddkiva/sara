// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#include <DO/Sara/Core/Image.hpp>

#include <numeric>
#include <queue>
#include <set>


namespace DO::Sara {

  enum class FastMarchingState : std::uint8_t
  {
    Alive = 0,
    Trial = 1,
    Far = 2,
    Forbidden = 3
  };


  template <typename T, int N>
  struct FastMarching
  {
    using value_type = T;
    using coords_type = Eigen::Matrix<int, N, 1>;

    struct CoordsVal
    {
      T val;
      coords_type coords;

      inline auto operator<(CoordsVal& other) const
      {
        return val < other.val;
      }
    };

    struct ImageBasedGraph
    {
      ImageBasedGraph() = default;
      ImageBasedGraph(const coords_type& sizes)
        : states{sizes}
        , distances{sizes}
        , predecessors{sizes}
      {
      }

      Image<FastMarchingState, N> states;
      Image<T, N> distances;
      Image<std::optional<coords_type>, N> predecessors;
    };

    using coords_val_type = CoordsVal;
    using image_based_graph_type = ImageBasedGraph;
    using trial_container_type = std::set<coords_val_type>;
    using alive_container_type = std::set<coords_val_type>;

    //! @brief Time complexity: O(V)
    FastMarching() = default;

    FastMarching(const coords_type& sizes, T limit)
      : g{sizes}
      , limit{limit}
    {
      // Initialize all vertices.
      g.states.fill(FastMarchingState::Far);
      g.distances.fill(std::numeric_limits<T>::max());
      g.predecessors.fill(std::nullopt);
    }

    //! @brief Initialize the data points
    auto initialize_alive_points(const std::vector<coords_type>& points) -> void
    {
      for (const auto& p : points)
        alive.insert(p);
    }

    //! @brief Initialize the trial points to bootstrap the fast marching.
    auto initialize_trial_queue() -> void
    {
      for (const auto& p : alive)
      {
        for (auto i = 0; i < N; ++i)
        {
          const coords_type n1 = p - coords_type::Unit(i);
          const coords_type n2 = p + coords_type::Unit(i);

          if (alive.find(n1) == alive.end())
            trial.insert(n1);
          if (alive.find(n2) == alive.end())
            trial.insert(n2);
        }
      }
    }

    //! @brief Time complexity: O(1). (Reusing Dijkstra terminology in CLRS
    //! book).
    auto relax(const coords_type& u, coords_type& v, image_based_graph_type& g)
        -> void
    {
      const auto dv_candidate = g.distances[u] + w(u, v);
      if (g.distances[v] < dv_candidate)
      {
        g.distances[v] = dv_candidate;
        g.predecessors[v] = u;
      }
    }

    //! @brief Time complexity: O(1). (Reusing Dijkstra terminology in CLRS
    //! book).
    auto extract_min()
    {
      auto min_it = std::begin(trial);
      const auto min = *min_it;
      trial.erase(min_it);
      return min;
    }

    auto increase_priority(const coords_val_type& v) -> void
    {
      if (g.distances[v.coords] < v.value)
      {
        const auto v_it = trial.find(v);
        trial.erase(v_it);
        trial.emplace(v.coords, g.distances[v.coords]);
      }
    }

    //! @brief The algorithm (Dijsktra).
    auto run() -> void
    {
      initialize_trial_queue();

      while (!trial.empty())
      {
        // Extract min.
        const auto u = extract_min();

        // Update its state.
        alive.push_back(u);
        g.states[u.coords] = FastMarchingState::Alive;

        // @TODO: check if we reach the limit.

        // Update the neighbors.
        for (auto& v : g.neighbors(u))
        {
          if (g.states[v] == FastMarchingState::Alive ||
              g.states[v] == FastMarchingState::Forbidden)
            continue;

          // From there, a point is either `Far` or `Trial` now.
          //
          // Update its distance value in both cases.
          relax(u.coords, v.coords, g);

          if (g.states[v] == FastMarchingState::Far)
          {
            // Update its state.
            g.states[v.coords] = FastMarchingState::Trial;

            // Insert it into the list of trial points.
            trial.emplace(v.coords, g.distance[v.coords]);
          }

          // Increase the priority of the point if necessary.
          if (g.states[v.coords] == FastMarchingState::Trial)
            increase_priority(v);
        }
      }
    }

    image_based_graph_type g;

    // Dijkstra algorithm.
    trial_container_type trial;
    alive_container_type alive;
    T limit;
  };

}  // namespace DO::Sara
