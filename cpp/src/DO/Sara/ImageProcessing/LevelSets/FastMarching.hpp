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


  template <int N>
  constexpr int pow(int x)
  {
    if constexpr (N == 0)
      return 1;
    else
      return x * pow<N - 1>(x);
  }

  static_assert(pow<2>(3) - 1 == 8);

  template <typename T, int N>
  struct FastMarching
  {
    struct CoordsValue
    {
      Matrix<int, N, 1> coords;
      T value;

      inline auto operator<(const CoordsValue& other) const
      {
        return value < other.value;
      }
    };

    using value_type = T;
    using coords_type = Eigen::Matrix<int, N, 1>;
    using coords_val_type = CoordsValue;
    using trial_set_type = std::multiset<coords_val_type>;

    std::array<coords_type, pow<N>(3) - 1> deltas = {
        Eigen::Vector2i{-1, 0},   //
        Eigen::Vector2i{+1, 0},   //
        Eigen::Vector2i{0, -1},   //
        Eigen::Vector2i{0, +1},   //
        Eigen::Vector2i{-1, -1},  //
        Eigen::Vector2i{-1, +1},  //
        Eigen::Vector2i{+1, -1},  //
        Eigen::Vector2i{+1, +1}   //
    };

    //! @brief Time complexity: O(V)
    FastMarching(const ImageView<T, N>& displacements,
                 T limit = std::numeric_limits<T>::max())
      : _displacements(displacements)
      , _states{displacements.sizes()}
      , _distances{displacements.sizes()}
      , _predecessors{displacements.sizes()}
      , _limit{limit}
    {
      reset();
    }

    //! @brief Reset the fast marching state.
    inline auto reset()
    {
      _states.flat_array().fill(FastMarchingState::Far);
      _distances.flat_array().fill(std::numeric_limits<T>::max());
      _predecessors.flat_array().fill(-1);
    }

    //! @brief Bootstrap the fast marching.
    inline auto initialize_alive_points(const std::vector<coords_type>& points) -> void
    {
      // Initialize the alive points to bootstrap the fast marching.
      for (const auto& p : points)
        _states(p) = FastMarchingState::Alive;

      // Initialize the trial points to bootstrap the fast marching.
      for (const auto& p : points)
      {
        for (const auto& delta : deltas)
        {
          const Eigen::Vector2i n = p + delta;
          if (n.x() < _margin.x() ||
              n.x() >= _displacements.width() - _margin.x() ||  //
              n.y() < _margin.y() ||
              n.y() >= _displacements.height() - _margin.y())
            continue;

          if (_states(n) == FastMarchingState::Alive ||
              _states(n) == FastMarchingState::Forbidden)
            continue;

          _states(n) = FastMarchingState::Trial;
          _distances(n) = _displacements(n);
          _predecessors(n) = to_index(p);

          _trial_set.insert({n, _distances(n)});
        }
      }
    }

    //! @brief The algorithm (Dijsktra).
    inline auto run() -> void
    {
      while (!_trial_set.empty())
      {
        // Extract the closest trial point.
        const auto p = _trial_set.begin()->coords;
        _trial_set.erase(_trial_set.begin());

        if (_states(p) == FastMarchingState::Alive)
        {
          std::cout << "OOPS!!!" << std::endl;
          continue;
        }

        // Update the neighbors.
        for (const auto& delta : deltas)
        {
          const coords_type n = p + delta;
          if (n.x() < _margin.x() ||
              n.x() >= _displacements.width() - _margin.x() ||  //
              n.y() < _margin.y() ||
              n.y() >= _displacements.height() - _margin.y())
            continue;

          if (_states(n) == FastMarchingState::Alive ||
              _states(n) == FastMarchingState::Forbidden)
            continue;

          // At this point, a neighbor is either `Far` or `Trial` now.
          //
          // Update its distance value in both cases.
          const auto new_dist_n =
              solve_eikonal_equation_2d(n, _displacements(n), _distances);
          if (new_dist_n < _distances(n))
          {
            _distances(n) = new_dist_n;
            _predecessors(n) = to_index(p);
          }

          if (_states(n) == FastMarchingState::Far)
          {
            // Update its state.
            _states(n) = FastMarchingState::Trial;

            // Insert it into the list of trial points.
            _trial_set.insert({n, _distances(n)});
          }

          // Increase the priority of the point if necessary.
          if (_states(n) == FastMarchingState::Trial)
            increase_priority(n, _distances(n));
        }
      }
    }

    inline auto to_index(const Eigen::Vector2i& p) const -> std::int32_t
    {
      return p.y() * _displacements.width() + p.x();
    }

    inline auto to_coords(const std::int32_t i) const -> Eigen::Vector2i
    {
      const auto y = i / _displacements.width();
      const auto x = i - y * _displacements.width();
      return {x, y};
    }

    inline auto solve_eikonal_equation_2d(const Eigen::Vector2i& x, const T fx,
                                          const Image<T>& u) -> T
    {
      const auto u0 = std::min(u(x - Eigen::Vector2i::Unit(0)),
                               u(x + Eigen::Vector2i::Unit(0)));
      const auto u1 = std::min(u(x - Eigen::Vector2i::Unit(1)),
                               u(x + Eigen::Vector2i::Unit(1)));

      const auto fx_inverse = 1 / (fx * fx);

      const auto a = 2.f;
      const auto b = -2 * (u0 + u1);
      const auto c = u0 * u0 + u1 * u1 - fx_inverse * fx_inverse;

      const auto delta = b * b - 4 * a * c;

      auto r0 = float{};
      if (delta >= 0)
        r0 = (-b + std::sqrt(delta)) / (2 * a);
      else
        r0 = std::min(u0, u1) + fx_inverse;

      return r0;
    }

    inline auto increase_priority(const coords_type& p, T value) -> void
    {
      if (value < _distances(p))
      {
        const auto p_it = _trial_set.find({p, _distances(p)});
        if (p_it != _trial_set.end() && p_it->coords == p)
          _trial_set.erase(p_it);
        _trial_set.insert({p, value});
      }
    }

    const ImageView<T, N>& _displacements;
    Image<FastMarchingState, N> _states;
    Image<T, N> _distances;
    Image<std::int32_t, N> _predecessors;
    coords_type _margin = coords_type::Ones();
    trial_set_type _trial_set;
    T _limit;
  };

}  // namespace DO::Sara
