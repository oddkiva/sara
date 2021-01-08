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

#include <DO/Sara/Core/Image/Image.hpp>
#include <DO/Sara/ImageProcessing/LevelSets/FastMarching.hpp>
// #include <DO/Sara/ImageProcessing/LevelSets/EikonalEquationSolver.hpp>


namespace DO::Sara {

  template <typename T, int N>
  class NarrowBand
  {
    ImageView<T, N>& _curr_level_set_func;
    Image<T, N> _prev_level_set_func;
    Image<bool, N> _band_map;

    using coords_type = Matrix<int, N, 1>;
    struct NarrowBandIterator;

    EikonalEquationSolver<T, N, +1> _reinit1;
    EikonalEquationSolver<T, N, -1> _reinit2;

  public:
    NarrowBand(ImageView<T, N>& phi)
      : _curr_level_set_func{phi}
      , _prev_level_set_func{phi.sizes()}
      , _band_map{phi.sizes()}
      , _reinit1(phi)
      , _reinit2(phi)
    {
    }

    using iterator = NarrowBandIterator;
    using const_iterator = iterator;

    iterator begin() const
    {
      return iterator(coords_type::Zero(), _band_map.sizes(), *this);
    }

    iterator end() const
    {
      return iterator(*this);
    }

    auto reinit_needed(T threshold) const -> bool
    {
      for (iterator p = begin(); p != end(); ++p)
      {
        const size_t o = _curr_level_set_func.offset(*p);
        const T v = _curr_level_set_func[o];
        const T w = _prev_level_set_func[o];

        if (w > threshold && v <= 0)
          return true;
        if (w < -threshold && v >= 0)
          return true;
      }

      return false;
    }

    auto insert(const coords_type& p) -> void
    {
      (*this)(p) = true;
    }

    auto clear_band() -> void
    {
      _band_map->fill(false);
    }

    template <class Approximator, class Integrator>
    void init(T thickness, Integrator& integr, int iter = 0, float dt = 0.4)
    {
      // Initializing fast-marching objects.
      _reinit1.init();
      _reinit2.init();

      clear_band();

      // Alive points are neighboring points of the zero level set.
      for (auto p = _curr_level_set_func.begin_array(); p.end(); ++p)
      {
        const T _phi = I(*p);
        coords_type pp, mp;

        for (int i = 0; i < N; i++)
        {
          pp = *p;
          mp = *p;
          if (mp[i] > 0)
            --mp[i];
          if (pp[i] < _curr_level_set_func.size(i) - 1)
            ++pp[i];

          if (_phi * _curr_level_set_func(pp) <= 0 ||
              _phi * _curr_level_set_func(mp) <= 0)
          {
            add_alive_point(*p);
            break;
          }
        }
      }

      // Trial points are neighbors of alive points
      _reinit1.initialize_trial_queue();
      _reinit2.initialize_trial_queue();

      // Remove points in the band that are neither trial or alive ones.
      // Set their values to infinite.
      for (auto p = _curr_level_set_func.begin_array(); !p.end(); ++p)
      {
        const size_t o = _curr_level_set_func.offset(*p);
        const T _phi = _curr_level_set_func[o];
        if (_phi > 0)
        {
          if (_reinit1.getState(o) == FastMarchingState::Far)
            _curr_level_set_func[o] = T(2);
          else
            (*this)[o] = true;
        }
        else
        {
          if (_reinit2.getState(o) == FastMarchingState::Far)
            _curr_level_set_func[o] = T(-2);
          else
            (*this)[o] = false;
        }
      }

      // EDP de reinitialisation
      reinit_pde<Approximator>(integr, iter, dt);

      // Fast marching
      reinit_fast_marching(thickness);

      // Assign far point values with the maximum value and appropriate sign.
      for (auto p = _curr_level_set_func.begin_array(); p.end(); ++p)
      {
        const size_t o = _curr_level_set_func.offset(*p);
        const T _phi = _curr_level_set_func[o];
        if (_phi > 0 && _reinit1.getState(o) == FastMarchingState::Far)
          _curr_level_set_func[o] = thickness;
        else if (_phi < 0 && _reinit2.getState(o) == FastMarchingState::Far)
          _curr_level_set_func[o] = -thickness;
      }

      // Recopy the reinitialized data.
      _prev_level_set_func = _curr_level_set_func;
    }

    template <typename Approximator, typename Integrator>
    void reinit(T thickness, Integrator& integr, int iter = 2, T dt = 0.4)
    {
      // Reinitialization PDE.
      reinit_pde<Approximator>(integr, iter, dt);

      // Initializing fast-marching objects.
      _reinit1.init();
      _reinit2.init();

      // Alive points are neighboring points of the zero level set.
      for (iterator p = begin(); p != end(); ++p)
      {
        const T _phi = I(*p);
        coords_type pp, mp;
        for (int i = 0; i < N; i++)
        {
          pp = *p;
          mp = *p;
          if (mp[i] > 0)
            mp[i]--;
          if (pp[i] < I.size(i) - 1)
            pp[i]++;
          if (_phi * I(pp) <= 0 || _phi * I(mp) <= 0)
          {
            add_alive_point(*p);
            break;
          }
        }
      }

      // Trial points are neighbors of alive points
      _reinit1.initialize_trial_queue();
      _reinit2.initialize_trial_queue();

      // Remove points in the band that are neither trial or alive ones.
      // Set their values to infinite.
      for (iterator p = begin(); p != end(); ++p)
      {
        const auto o = _curr_level_set_func.offset(*p);
        const auto _phi = _curr_level_set_func[o];
        if (_phi > 0 && _reinit1.getState(o) == FastMarchingState::Far)
        {
          _curr_level_set_func[o] = thickness;
          _band_map[o] = false;
        }
        else if (_phi <= 0 && _reinit2.getState(o) == FastMarchingState::Far)
        {
          _curr_level_set_func[o] = -thickness;
          _band_map[o] = false;
        }
      }

      // Fast marching
      reinit_fast_marching(thickness);

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
      const T _phi = I(p);
      if (_phi > 0)
      {
        _reinit1.addAlivePoint(p);
        _reinit2.addForbiddenPoint(p);
      }
      else
      {
        _reinit1.addForbiddenPoint(p);
        _reinit2.addAlivePoint(p);
      }
    }

    template <typename Approximator, typename Integrator>
    auto reinit_pde(Integrator& integr, int iter = 2, T dt = 0.4) -> void
    {
      for (int t = 0; t < iter; t++)
      {
        do
        {
          for (iterator p = begin(); p != end(); ++p)
            integr(*p) = reinitialization<Approximator>(I, *p);
        } while (!integr.step(begin(), end(), dt));
      }
    }

    auto reinit_fast_marching(T thickness, T keep = 0) -> void
    {
      // Perform fast marching in both directions.
      if (thickness > 0)
      {
        _reinit1.run(thickness);
        _reinit2.run(-thickness);
      }
      else
      {
        // Fast marching.
        _reinit1.run();
        _reinit2.run();
      }

      // Add alive points to the band as the fast marching is running through.
      using ConstCoordsListIterator = std::list<Coords<N>>::const_iterator;
      if (keep == 0)
      {
        for (ConstCoordsListIterator p = _reinit1.begin(); p != _reinit1.end();
             ++p)
          insert(*p);
        for (ConstCoordsListIterator p = _reinit2.begin(); p != _reinit2.end();
             ++p)
          insert(*p);
      }
      else
      {
        for (ConstCoordsListIterator p = _reinit1.begin(); p != _reinit1.end();
             ++p)
          if (I(*p) < keep)
            insert(*p);
        for (ConstCoordsListIterator p = _reinit2.begin(); p != _reinit2.end();
             ++p)
          if (I(*p) > -keep)
            insert(*p);
      }
    }

    auto expand_band(int expand) -> void
    {
      for (int k = 0; k < expand; k++)
      {
        for (iterator p = begin(); p != end(); ++p)
        {
          const coords_type pp = p - coords_type::Ones();
          const coords_type mp = p + coords_type::Ones();
          for (CoordsIterator<N> it(mp, pp); it != CoordsIterator<N>(); ++it)
            insert(*it);
        }
      }
    }

    auto shrink_band(T thickness) -> void
    {
      // Scan the points of the narrow band.
      for (iterator p = begin(); p != end(); ++p)
      {
        const int o = I.offset(*p);
        const T v = I[o];

        // Based on the signed distance value,
        if (v > thickness)
        {
          // Assign the maximum level set value.
          (*this)[o] = false;
          _prev_level_set_func[o] = I[o] = thickness;
        }
        else if (v < -thickness)
        {
          // Assign the minimum level set value.
          (*this)[o] = false;
          _prev_level_set_func[o] = I[o] = -thickness;
        }
        else
        {
          // The point is in the band, store its level set value.
          _prev_level_set_func[o] = v;
        }
      }
    }
  };

}  // namespace DO::Sara
