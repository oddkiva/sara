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


namespace DO::Sara {

  //! @{

  //! Forward Euler time integrator.
  template <typename T, int N>
  struct EulerIntegrator
  {
    Image<T, N> _df;  //!< velocity function.
    ImageView<T, N> _f;   //!< level set function.

  public:
    EulerIntegrator(ImageView<T, N>& f0)
      : _df{f0.sizes()}
      , _f{f0}
    {
    }

    //! @brief Update the function `f` on the specific domain and with time step
    //! `dt`.
    bool step(const ImageView<bool, N>& domain, T dt)
    {
      for (auto p = domain.begin_array(); !p.end(); ++p)
        if (*p)
          _f(p.position()) += dt * _df(p.position());
      return true;
    }
  };


  //! Midpoint time integrator.
  template <typename T, int N>
  struct MidpointIntegrator
  {
    Image<T, N> _df;
    Image<T, N> _f;

    Image<T, N> _midpoint;
    int substep = 0;

    MidpointIntegrator(ImageView<T, N>& f0)
      : _df{f0.sizes()}
      , _f{f0}
      , _midpoint{f0}
    {
    }

    bool step(const ImageView<bool, N>& domain, T dt)
    {
      // First substep
      if (substep == 0)
      {
        for (auto p = domain.begin_array(); !p.end(); ++p)
        {
          if (!*p)
            continue;
          _midpoint(p.coords()) = _f(p.coords());
          _f(p.coords()) += (dt / T(2)) * _df(p.coords()); // half the time step.
        }

        ++substep;
        return false;
      }

      // Second substep with the full time step.
      for (auto p = domain.begin_array(); !p.end(); ++p)
      {
        if (!*p)
          continue;
        _f(p.coords()) = _midpoint(p.coords()) + dt * _df(p.coords());
      }
      substep = 0;

      return true;
    }
  };

  //!  @}

}  // namespace DO::Sara
