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

#include <DO/Sara/Core/ArrayIterators/CoordinatesIterator.hpp>
#include <DO/Sara/Core/Image.hpp>
#ifdef VISUAL_INSPECTION
#include <DO/Sara/Graphics.hpp>
#endif

#include <array>
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
  constexpr auto pow(int x) -> int
  {
    if constexpr (N == 0)
      return 1;
    else if constexpr (N == 1)
      return x;
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
        // if (value < other.value)
        //   return true;
        // if (lexicographical_compare(coords, other.coords))
        //   return true;
        // return false;
      }
    };

    using value_type = T;
    using coords_type = Eigen::Matrix<int, N, 1>;
    using coords_val_type = CoordsValue;
    using trial_set_type = std::multiset<coords_val_type>;

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
    inline auto initialize_alive_points(const std::vector<coords_type>& points)
        -> void
    {
      // Initialize the alive points to bootstrap the fast marching.
      for (const auto& p : points)
      {
        _states(p) = FastMarchingState::Alive;
        _distances(p) = 0;
      }

      // Initialize the trial points to bootstrap the fast marching.
      initialize_trial_set_from_alive_set(points);
    }

    inline auto initialize_trial_set_from_alive_set(
        const std::vector<coords_type>& alive_set) -> void
    {
      // Initialize the trial points to bootstrap the fast marching.
      for (const auto& p : alive_set)
      {
        for (const auto& delta: _deltas)
        {
          const coords_type n = p + delta;
          auto in_image_domain = true;
          for (auto i = 0; i < N; ++i)
          {
            if (n(i) < _margin(i) ||
                n(i) >= _displacements.size(i) - _margin(i))
            {
              in_image_domain = false;
              break;
            }
          }
          if (!in_image_domain)
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
        const auto [p, val] = *_trial_set.begin();
        if (val > _limit)
          break;

        _trial_set.erase(_trial_set.begin());

#ifdef VISUAL_INSPECTION
        SARA_DEBUG << "p = " << p.transpose() << "   v = "
                   << _trial_set.begin()->value << std::endl;
        draw_point(p.x(), p.y(), Green8);
#endif

        if (_states(p) == FastMarchingState::Alive)
        {
          std::cout << "OOPS!!!" << std::endl;
          continue;
        }

        // Update the neighbors.
        for (const auto& delta: _deltas)
        {
          const coords_type n = p + delta;

          auto in_image_domain = true;
          for (auto i = 0; i < N; ++i)
          {
            if (n(i) < _margin(i) ||
                n(i) >= _displacements.size(i) - _margin(i))
            {
              in_image_domain = false;
              break;
            }
          }
          if (!in_image_domain)
            continue;

          if (_states(n) == FastMarchingState::Alive ||
              _states(n) == FastMarchingState::Forbidden)
            continue;

          // At this point, a neighbor is either `Far` or `Trial` now.
          //
          // Update its distance value in both cases.
          const auto new_dist_n = solve_eikonal_equation(n,
                                                         _displacements(n),
                                                         _distances);
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

    inline auto to_index(const coords_type& p) const -> std::int32_t
    {
      return p.dot(_displacements.strides());
    }

    inline auto to_coords(std::int32_t index) const -> coords_type
    {
      const auto c = coords_type{};
      const auto s = _displacements.strides();
      if (s[0] != 1)
        throw std::runtime_error{"Not Implemented!"};

      for (auto i = N - 1; i >= 0; ++i)
      {
        if (i == N - 1)
          c[i] = i / s[i];
        else
        {
          index -= c[i + 1] * s[i + 1];
          c[i] = index / s[i];
        }
      }
      return c;
    }

    //! @brief Solve the first order approximation of the Eikonal equation.
    //! This involves solving a second-degree polynomial.
    //!
    //! For dimension N >= 3, the implementation is understood from the
    //! Wikipedia article, so I hope it is correct.
    inline auto solve_eikonal_equation(const coords_type& x,      //
                                       const T fx,                //
                                       const ImageView<T, N>& u)  //
        -> T
    {
      auto us = Eigen::Matrix<T, N, 1>{};
      for (auto i = 0; i < N; ++i)
        us[i] = std::min(u(x - coords_type::Unit(i)),
                         u(x + coords_type::Unit(i)));
      const auto fx_inverse = 1 / fx;
      const auto usum = us.array().sum();

      // Calculate the reduced discriminant of the trinome we are solving.
      const auto delta = std::pow(us.sum(), 2) -
                         N * (us.squaredNorm() - std::pow(fx_inverse, 2));

      auto r0 = T{};
      if (delta >= 0)
        r0 = (usum + std::sqrt(delta)) / N;
      else
      {
        if constexpr (N == 2)
          r0 = us.minCoeff() + fx_inverse;
        else
        {
          const auto umin = find_min_coefficient(us);
          r0 = umin + fx_inverse;
        }
      }

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

    static inline auto find_min_coefficient(const Eigen::Matrix<T, N, 1>& us)
        -> T
    {
      auto umins = Eigen::Matrix<T, N - 1, N>{};
      for (auto j = 0; j < N; ++j)
      {
        if (j == 0)
          umins.col(j) << us.segment(1, N - 1);
        else if (j == N - 1)
          umins.col(j) << us.head(N - 1);
        else
          umins.col(j) << us.head(j), us.segment(j + 1, N - j - 1);
      }

      return umins.colwise().minCoeff().minCoeff();
    }

    static auto initialize_deltas_4() -> std::array<coords_type, 2 * N>
    {
      if constexpr (N == 2)
        return {
          // dim 0
          Eigen::Vector2i{-1,  0},  //
          Eigen::Vector2i{+1,  0},  //
          // dim 1
          Eigen::Vector2i{ 0, -1},  //
          Eigen::Vector2i{ 0, +1},  //
        };
      else
      {
        auto deltas = std::array<coords_type, 2 * N>{};
        for (auto i = 0; i < N; ++i)
          for (auto s : {-1, 1})
            deltas.push_back(coords_type::Zero() + s * coords_type::Unit(i));

        return deltas;
      }
    }

    static auto initialize_deltas_8() -> std::array<coords_type, pow<N>(3) - 1>
    {
      if constexpr (N == 2)
        return {
          // row -1
          Eigen::Vector2i{-1, -1},  //
          Eigen::Vector2i{-1,  0},  //
          Eigen::Vector2i{+1,  0},  //
          // row  0
          Eigen::Vector2i{ 0, -1},  //
          Eigen::Vector2i{ 0, +1},  //
          // row +1
          Eigen::Vector2i{+1, -1},  //
          Eigen::Vector2i{+1, +1},  //
          Eigen::Vector2i{-1, +1}   //
        };
      else
      {
        auto deltas = std::array<coords_type, pow<N>(3) - 1>{};
        const coords_type start = -coords_type::Ones();
        const coords_type end = start + 3 * coords_type::Ones();
        using coords_iterator = CoordinatesIterator<ImageView<T, N>>;

        auto i = 0;
        for (auto c = coords_iterator{start, end}; !c.end(); ++c)
        {
          if (*c == coords_type::Zero())
            continue;
          deltas[i++] = *c;
        }

        return deltas;
      }
    }

    // const std::array<coords_type, 2 * N> _deltas = initialize_deltas_4();
    const std::array<coords_type, pow<N>(3) - 1> _deltas = initialize_deltas_8();

    const ImageView<T, N> _displacements;
    Image<FastMarchingState, N> _states;
    Image<T, N> _distances;
    Image<std::int32_t, N> _predecessors;
    coords_type _margin = coords_type::Ones();
    trial_set_type _trial_set;
    T _limit;
  };

}  // namespace DO::Sara
