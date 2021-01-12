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
  class EulerIntegrator
  {
  private:
    ImageView<T, N>& _df;
    Image<T, N> _f;

  public:
    EulerIntegrator(ImageView<T, N>& df)
      : _df{df}
      , _f{df.sizes()}
    {
    }

    template <typename CoordsIterator>
    bool step(CoordsIterator begin, CoordsIterator end, T dt)
    {
      for (auto p = begin; p != end; ++p)
        _f(p.coords()) += dt * _df(p.coords());
      return true;
    }
  };


  //! Midpoint time integrator.
  template <typename T, int N>
  class MidpointIntegrator
  {
  private:
    ImageView<T, N>& _df;
    Image<T, N> _f;

    Image<T, N> _midpoint;
    int substep = 0;

  public:
    MidpointIntegrator(ImageView<T, N>& df)
      : _df{df}
      , _f{df}
      , _midpoint{df}
    {
    }

    template <typename CoordsIterator>
    bool step(CoordsIterator begin, CoordsIterator end, T dt)
    {
      if (substep == 0)
      {
        for (auto p = begin; p != end; ++p)
        {
          _midpoint(p.coords()) = _f(p.coords());
          _f(p.coords()) += (dt / T(2)) * _df(p.coords());
        }

        ++substep;
        return false;
      }

      // Second substep
      for (auto p = begin; p != end; ++p)
        _f(p.coords()) = _midpoint(p.coords()) + dt * _df(p.coords());

      substep = 0;

      return true;
    }
  };

  // @TODO: Runge-Kutta RK4 integrator.
  //!  @}

}  // namespace DO::Sara
