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

    FastMarching<T, N> _exterior_reinitializer;
    FastMarching<T, N> _interior_reinitializer;

  public:
    NarrowBand(ImageView<T, N>& phi)
      : _curr_level_set_func{phi}
      , _prev_level_set_func{phi.sizes()}
      , _band_map{phi.sizes()}
      , _exterior_reinitializer(phi)
      , _interior_reinitializer(phi)
    {
      _zeros.reserve(phi.size());
    }

    //! @ brief Check the temporal coherence of the sign of the level set
    //! function.
    /*!
     *  We need to ensure high level set values cannot flip their sign all of
     *  sudden during an iteration.
     */
    auto reinit_needed(T threshold) const -> bool
    {
      auto band_map_flat = _band_map.flat_array();
      auto curr_flat = _curr_level_set_func.flat_array();
      auto prev_flat = _prev_level_set_func.flat_array();

      for (auto p = 0u; p != _band_map.size(); ++p)
      {
        if (!band_map_flat(p))
          continue;

        const auto phi_curr = curr_flat(p);
        const auto phi_prev = prev_flat(p);

        if (phi_prev > threshold && phi_curr <= 0)
          return true;
        if (phi_prev < -threshold && phi_curr >= 0)
          return true;
      }

      return false;
    }

    //! @brief Populate the coordinates of level zero-crossing in the level
    //! set.
    auto populate_zero_crossings(std::vector<coords_type>& zeros)
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
      _exterior_reinitializer.reset();
      _interior_reinitializer.reset();

      // Bootstrap the fast marching method.
      for (const auto& p : zeros)
      {
        const auto& _phi = _curr_level_set_func(p);
        if (_phi > 0)
        {
          _exterior_reinitializer._states(p) = FastMarchingState::Alive;
          _interior_reinitializer._states(p) = FastMarchingState::Forbidden;
        }
        else
        {
          _exterior_reinitializer._states(p) = FastMarchingState::Forbidden;
          _interior_reinitializer._states(p) = FastMarchingState::Alive;
        }
      }

      // Manually initialize the trial queues.
      _exterior_reinitializer.initialize_trial_set_from_alive_set(zeros);
      _interior_reinitializer.initialize_trial_set_from_alive_set(zeros);
    }

    template <typename Approximator, typename Integrator>
    auto init(T thickness, Integrator& integr, int iter = 0, float dt = 0.4)
        -> void
    {
      // Set the band as the empty set.
      reset_band();

      // Locate the zero crossings of the level set function.
      populate_zero_crossings(_zeros);
      initialize_fast_marching(_zeros);

      // Initialize the band as the set of points that are either trial or
      // alive.
      //
      // Set their values to +/-2 due to boundary conditions of PDE.
      for (auto phi_p = _curr_level_set_func.begin_array(); !phi_p.end();
           ++phi_p)
      {
        const auto& p = phi_p.position();

        if (*phi_p > 0)  // We are outside the domain.
        {
          if (_exterior_reinitializer._states(p) == FastMarchingState::Far)
            *phi_p = T(2);
          else
            _band_map(p) = true;
        }
        else  // We are inside the domain enclosed by the shape.
        {
          if (_interior_reinitializer._states(p) == FastMarchingState::Far)
            *phi_p = T(-2);
          else
            _band_map(p) = true;
        }
      }

      reinitialize_level_set_function<Approximator>(integr, iter, dt);

      // Calculate the new signed distance functions.
      run_fast_marching_sideways(thickness);

      // Assign far point values with the maximum value and appropriate sign.
      for (auto phi_p = _curr_level_set_func.begin_array(); phi_p.end();
           ++phi_p)
      {
        const auto& p = phi_p.position();
        if (*phi_p > 0 &&
            _exterior_reinitializer._states(p) == FastMarchingState::Far)
          *phi_p = thickness;
        else if (*phi_p < 0 &&
                 _interior_reinitializer._states(p) == FastMarchingState::Far)
          *phi_p = -thickness;
      }

      _prev_level_set_func = _curr_level_set_func;
    }

    template <typename Approximator, typename Integrator>
    auto reinit(T thickness, Integrator& integr, int iter = 2, T dt = 0.4)
        -> void
    {
      // Reinitialization the level set function.
      reinitialize_level_set_function<Approximator>(integr, iter, dt);

      // Locate the zero crossings of the level set function.
      populate_zero_crossings(_zeros);
      initialize_fast_marching(_zeros);

      // Remove points in the band that are neither trial or alive.
      // Set their values to the the maximum band thickness.
      for (auto p = _band_map.begin_array(); !p.end(); ++p)
      {
        if (!(*p))
          continue;

        const auto phi = _curr_level_set_func(p.position());
        if (phi > 0 &&  //
            _exterior_reinitializer._states(p.position()) ==
                FastMarchingState::Far)
        {
          _curr_level_set_func(p.position()) = thickness;
          *p = false;
        }
        else if (phi <= 0 &&  //
                 _interior_reinitializer._states(p.position()) ==
                     FastMarchingState::Far)
        {
          _curr_level_set_func(p.position()) = -thickness;
          *p = false;
        }
      }

      // Compute the distance function.
      run_fast_marching_sideways(thickness);

      // Recopy the reinitialized data.
      _prev_level_set_func = _curr_level_set_func;
    }

    //! @brief Maintain the signed distance.
    template <typename Approximator, typename Integrator>
    auto maintain(T thickness, Integrator& integr, int expand = 1, int iter = 1,
                  T dt = 0.4)  //
        -> void
    {
      expand_band(expand);
      reinitialize_level_set_function<Approximator>(integr, iter, dt);
      shrink_band(thickness);
    }

    //! @brief Reinitialize the level set function with the same zero level set
    //! but with |∇Φ| = 1.
    /*!
     *  This is for numerical stability reasons and we use the time integration
     *  approach for a few time steps.
     *
     *  cf. https://math.mit.edu/classes/18.086/2008/levelsetpres.pdf
     */
    template <typename Approximator, typename TimeIntegrator>
    auto reinitialize_level_set_function(TimeIntegrator& integrator,
                                         int iter = 2, T dt = 0.4) -> void
    {
      for (auto t = 0; t < iter; ++t)
      {
        // Here we make sure we perform a full step and not substep.
        // In particular for the midpoint time integrator.
        do
        {
          for (auto p = _curr_level_set_func.begin_array(); !p.end(); ++p)
            integrator._f(p.position()) = reinitialization<Approximator>(  //
                _curr_level_set_func,                                   //
                p.position()                                            //
            );
        } while (!integrator.step(_band_map, dt));
      }
    }

    //! @brief Calculate the signed distances.
    auto run_fast_marching_sideways(T thickness, T keep = 0) -> void
    {
      // Perform the fast marching sideways.
      thickness = thickness > 0 ? thickness: std::numeric_limits<T>::max();
      _exterior_reinitializer._limit = thickness;
      _interior_reinitializer._limit = -thickness;
      _exterior_reinitializer.run();
      _interior_reinitializer.run();

      // The band is the set of alive points whose level set value is in the
      // range ]-keep, keep[.
      if (keep == 0)
      {
        // TODO: use std::transform instead.
        for (auto p = _exterior_reinitializer._states.begin_array(); !p.end();
             ++p)
          if (*p == FastMarchingState::Alive)
            _band_map(p.position()) = true;
        for (auto p = _interior_reinitializer._states.begin_array(); !p.end();
             ++p)
          if (*p == FastMarchingState::Alive)
            _band_map(p.position()) = true;
      }
      else
      {
        // TODO: use std::transform instead.
        for (auto p = _exterior_reinitializer._states.begin_array(); !p.end();
             ++p)
        {
          if (*p != FastMarchingState::Alive)
            continue;
          if (_curr_level_set_func(p.position()) < keep)
            _band_map(p.position()) = true;
        }

        for (auto p = _interior_reinitializer._states.begin_array(); !p.end();
             ++p)
        {
          if (*p != FastMarchingState::Alive)
            continue;
          if (_curr_level_set_func(p.position()) > -keep)
            _band_map(p.position()) = true;
        }
      }
    }

    //! @brief Reset the band as an empty set.
    auto reset_band() -> void
    {
      _band_map.flat_array().fill(false);
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
