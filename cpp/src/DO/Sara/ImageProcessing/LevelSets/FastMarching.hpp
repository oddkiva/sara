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
#include <set>


namespace DO::Sara {


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

    Image<State, N> states;
    Image<T, N> distances;
    Image<std::optional<coords_type>, N> predecessors;
  };


  template <typename T, int N, typename WeightFn>
  struct FastMarching
  {
    using value_type = T;
    using image_based_graph_type = ImageBasedGraph<T, N>;
    using coords_type = typename image_based_graph_type::coords_type;
    using weight_function_type = WeightFn;

    struct CoordsVal
    {
      T val;
      coords_type coords;

      inline auto operator<(CoordsVal& other) const
      {
        return val < other.val;
      }
    };

    using coords_val_type = CoordsVal;
    using trial_container_type = std::set<coords_val_type>;
    using alive_container_type = std::set<coords_val_type>;

    //! @brief Time complexity: O(V)
    void initialize()
    {
      // Initialize all vertices.
      g.states.fill(State::Far);
      g.distances.fill(std::numeric_limits<T>::max());
      g.predecessors.fill(std::nullopt);

      trial = trial_container_type{};
      alive = alive_container_type{};
    }

    //! @brief Time complexity: O(1). (Reusing Dijkstra terminology in CLRS
    //! book).
    void relax(const coords_type& u, coords_type& v, image_based_graph_type& g)
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
#ifdef DEBUG
      if (trial.empty())
        throw std::runtime_error("List is empty!");
#endif
      auto min_it = std::begin(trial);
      const auto min = *min_it;
      trial.erase(min_it);
      return min;
    }

    auto increase_priority(const coords_val_type& v)
    {
      if (g.distances[v.coords] < v.value)
      {
        const auto v_it = trial.find(v);
        trial.erase(v_it);
        trial.emplace(v.coords, g.distances[v.coords]);
      }
    }

    void operator()() const
    {
      initialize();

      // Initialize with trial points.
      // for (auto v : trial_list)
      //  trial.push_back(...);
      // std::make_heap(std::begin(trial), std::end(trial));

      while (!trial.empty())
      {
        // Extract min.
        const auto u = extract_min();

        // Update its state.
        alive.push_back(u);
        g.states[u.coords] = State::Alive;

        // @TODO: check if we reach the limit.

        // Update the neighbors.
        for (auto& v : g.neighbors(u))
        {
          if (g.states[v] == State::Alive || g.states[v] == State::Forbidden)
            continue;

          // From there, a point is either `Far` or `Trial` now.
          //
          // Update its distance value in both cases.
          relax(u.coords, v.coords, g);

          if (g.states[v] == State::Far)
          {
            // Update its state.
            g.states[v.coords] = State::Trial;

            // Insert it into the list of trial points.
            trial.emplace(v.coords, g.distance[v.coords]);
          }

          // Increase the priority of the point if necessary.
          if (g.states[v.coords] == State::Trial)
            increase_priority(v);
        }
      }
    }

    image_based_graph_type g;

    // Weight function typically the Eikonal equation.
    weight_function_type w;

    // Dijkstra algorithm.
    trial_container_type trial;
    alive_container_type alive;
    T limit;
  };

}  // namespace DO::Sara
