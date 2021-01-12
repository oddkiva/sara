// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/ImageProcessing/LevelSets/FastMarching.hpp>
#include <DO/Sara/ImageProcessing/LevelSets/Flux.hpp>


namespace DO::Sara {

  template <typename T, int N>
  struct NarrowBand
  {
    ImageView<T, N>& _curr_level_set_func;
    Image<T, N> _prev_level_set_func;
    Image<bool, N> _band_map;

    using coords_type = Matrix<int, N, 1>;
    using coords_iterator = CoordinatesIterator<ImageView<T, N>>;

    std::vector<coords_type> _zeros;

    FastMarching<T, N> _reinit1;
    FastMarching<T, N> _reinit2;

  public:
    NarrowBand(ImageView<T, N>& phi)
      : _curr_level_set_func{phi}
      , _prev_level_set_func{phi.sizes()}
      , _band_map{phi.sizes()}
      , _reinit1(phi)
      , _reinit2(phi)
    {
      _zeros.reserve(phi.size());
    }

    auto reinit_needed(T threshold) const -> bool
    {
      auto band_map_flat = _band_map.flat_array();
      auto curr_flat = _curr_level_set_func.flat_array();
      auto prev_flat = _prev_level_set_func.flat_array();
      for (auto p = 0; p != _band_map.size(); ++p)
      {
        if (!band_map_flat(p))
          continue;

        const auto v = curr_flat(p);
        const auto w = prev_flat(p);

        if (w > threshold && v <= 0)
          return true;
        if (w < -threshold && v >= 0)
          return true;
      }

      return false;
    }

    auto reset_band() -> void
    {
      _band_map->fill(false);
    }

    auto locate_zeros(std::vector<coords_type>& zeros)
    {
      zeros.clear();
      for (auto phi_p = _curr_level_set_func.begin_array(); phi_p.end();
           ++phi_p)
      {
        const auto& curr = phi_p.position();
        const auto phi_curr = *phi_p;

        for (auto i = 0; i < N; ++i)
        {
          auto next = curr;
          auto prev = curr;
          if (prev(i) > 0)
            --prev(i);
          if (next(i) < _curr_level_set_func.size(i) - 1)
            ++next(i);

          const auto phi_next = _curr_level_set_func(next);
          const auto phi_prev = _curr_level_set_func(prev);

          const auto zero_crossing = phi_curr * phi_next <= 0 ||  //
                                     phi_curr * phi_prev <= 0;
          if (zero_crossing)
          {
            zeros.push_back(curr);
            break;
          }
        }
      }
    }

    auto initialize_fast_marching(const std::vector<coords_type>& zeros) -> void
    {
      // Reset the fast marching states.
      _reinit1.reset();
      _reinit2.reset();

      // Bootstrap the fast marching method.
      for (auto phi_p = _curr_level_set_func.begin_array(); !phi_p.end();
           ++phi_p)
      {
        const auto& p = phi_p.position();
        if (*phi_p > 0)
        {
          _reinit1._states(p) = FastMarchingState::Alive;
          _reinit2._states(p) = FastMarchingState::Forbidden;
        }
        else
        {
          _reinit1._states(p) = FastMarchingState::Forbidden;
          _reinit2._states(p) = FastMarchingState::Alive;
        }
      }
    }

    template <class Approximator, class Integrator>
    void init(T thickness, Integrator& integr, int iter = 0, float dt = 0.4)
    {
      // Set the band as the empty set.
      reset_band();

      // Locate the zero crossings of the level set function.
      locate_zeros(_zeros);

      initialize_fast_marching(_zeros);

      // Initialize the band as the set of points that are either trial or
      // alive.
      //
      // Set their values to +/-2 due to boundary conditions of PDE.
      for (auto phi_p = _curr_level_set_func.begin_array(); !phi_p.end();
           ++phi_p)
      {
        const auto& p = phi_p.position();
        if (*phi_p > 0)
        {
          // We are in the exterior of the shape.
          if (_reinit1._states(p) == FastMarchingState::Far)
            *phi_p = T(2);
          else
            _band_map(p) = true;
        }
        else  // phi_p <= 0
        {
          // We are in the interior of the shape.
          if (_reinit2._states(p) == FastMarchingState::Far)
            *phi_p = T(-2);
          else
            _band_map(p) = true;
        }
      }

      reinit_pde<Approximator>(integr, iter, dt);

      run_fast_marching_sideways(thickness);

      // Assign far point values with the maximum value and appropriate sign.
      for (auto phi_p = _curr_level_set_func.begin_array(); phi_p.end(); ++phi_p)
      {
        const auto& p = phi_p.position();
        if (*phi_p > 0 && _reinit1._states(p) == FastMarchingState::Far)
          *p = thickness;
        else if (*phi_p < 0 && _reinit2._states(p) == FastMarchingState::Far)
          *p = -thickness;
      }

      _prev_level_set_func = _curr_level_set_func;
    }

    template <typename Approximator, typename Integrator>
    void reinit(T thickness, Integrator& integr, int iter = 2, T dt = 0.4)
    {
      // Reinitialization PDE.
      reinit_pde<Approximator>(integr, iter, dt);

      // Locate the zero crossings of the level set function.
      locate_zeros(_zeros);
      initialize_fast_marching(_zeros);

      // Remove points in the band that are neither trial or alive ones.
      // Set their values to infinite.
      for (auto p = _band_map.begin_array(); !p.end(); ++p)
      {
        if (!(*p))
          continue;

        const auto phi = _curr_level_set_func(p.position());
        if (phi > 0 &&
            _reinit1._states(p.position()) == FastMarchingState::Far)
        {
          _curr_level_set_func(p.position()) = thickness;
          *p = false;
        }
        else if (phi <= 0 &&
                 _reinit2._states(p.position()) == FastMarchingState::Far)
        {
          _curr_level_set_func(p.position()) = -thickness;
          *p = false;
        }
      }

      // Fast marching
      run_fast_marching_sideways(thickness);

      // Recopy the reinitialized data.
      _prev_level_set_func = _curr_level_set_func;
    }

    // Maintain the signed distance.
    template <typename Approximator, typename Integrator>
    auto maintain(T thickness, Integrator& integr, int expand = 1, int iter = 1,
                  T dt = 0.4) -> void
    {
      expand_band(expand);
      reinit_pde<Approximator>(integr, iter, dt);
      shrink_band(thickness);
    }

    auto add_alive_point(const coords_type& p) -> void
    {
      const auto& _phi = I(p);
      if (_phi > 0)
      {
        _reinit1._states(p) = FastMarchingState::Alive;
        _reinit2._states(p) = FastMarchingState::Forbidden;
      }
      else
      {
        _reinit1._states(p) = FastMarchingState::Forbidden;
        _reinit2._states(p) = FastMarchingState::Alive;
      }
    }

    // Update the level set function using the time integrator.
    template <typename Approximator, typename TimeIntegrator>
    auto reinit_pde(TimeIntegrator& integrator, int iter = 2, T dt = 0.4)
        -> void
    {
      for (auto t = 0; t < iter; ++t)
      {
        do
        {
          // Reinitialization PDE.
          for (auto p = _curr_level_set_func.begin_array(); !p.end(); ++p)
            integrator(p.position()) = reinitialization<Approximator>(  //
                _curr_level_set_func,                                   //
                p.position()                                            //
            );
        } while (!integrator.step(_band_map.begin_array(), dt));
      }
    }

    auto run_fast_marching_sideways(T thickness, T keep = 0) -> void
    {
      // Perform the fast marching sideways.
      if (thickness > 0)
      {
        _reinit1.run(thickness);
        _reinit2.run(-thickness);
      }
      else
      {
        _reinit1.run();
        _reinit2.run();
      }

      // The band is the set of alive points whose level set value is in the
      // range ]-keep, keep[.
      if (keep == 0)
      {
        // TODO: use std::transform instead.
        for (auto p = _reinit1._states.begin_array(); !p.end(); ++p)
          if (*p == FastMarchingState::Alive)
            _band_map(p.position()) = true;
        for (auto p = _reinit2._states.begin_array(); !p.end(); ++p)
          if (*p == FastMarchingState::Alive)
            _band_map(p.position()) = true;
      }
      else
      {
        // TODO: use std::transform instead.
        for (auto p = _reinit1._states.begin_array(); !p.end(); ++p)
        {
          if (*p != FastMarchingState::Alive)
            continue;
          if (_curr_level_set_func(p.position()) < keep)
            _band_map(p.position()) = true;
        }

        for (auto p = _reinit2._states.begin_array(); !p.end(); ++p)
        {
          if (*p != FastMarchingState::Alive)
            continue;
          if (_curr_level_set_func(p.position()) > -keep)
            _band_map(p.position()) = true;
        }
      }
    }

    auto expand_band(int expand) -> void
    {
      for (int k = 0; k < expand; k++)
      {
        for (auto p = _band_map.begin_array(); !p.end(); ++p)
        {
          if (!(*p))
            continue;

          // Add the neighbors of the band.
          const coords_type pp = p - coords_type::Ones();
          const coords_type mp = pp + 3 * coords_type::Ones();
          for (auto it = coords_iterator{mp, pp}; it.end(); ++it)
            _band_map(*it) = true;
        }
      }
    }

    auto shrink_band(T thickness) -> void
    {
      // Scan the points of the narrow band.
      for (auto p = _band_map.begin_array(); !p.end(); ++p)
      {
        if (!(*p))
          continue;

        const T v = _curr_level_set_func(p.position());

        // Based on the signed distance value,
        if (v > thickness)
        {
          // Assign the maximum level set value.
          *p = false;
          _curr_level_set_func(p.position()) = thickness;
          _prev_level_set_func(p.position()) = thickness;
        }
        else if (v < -thickness)
        {
          // Assign the minimum level set value.
          *p = false;
          _curr_level_set_func(p.position()) = -thickness;
          _prev_level_set_func(p.position()) = -thickness;
        }
        else
        {
          // The point is in the band, store its level set value.
          _prev_level_set_func(p.position()) = v;
        }
      }
    }
  };

}  // namespace DO::Sara
