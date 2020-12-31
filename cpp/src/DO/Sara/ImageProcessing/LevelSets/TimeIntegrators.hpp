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

  //! First-order one-step Euler time integrator
  template <typename T, int N>
  class Euler : public ImageView<T, N>
  {
  private:
    ImageView<T, N>& I;

  public:
    Euler(Image<T, N>& _I)
      : Image<T, N>(_I.sizes())
      , I(_I)
    {
    }

    template <typename InputIterator>
    bool step(InputIterator first, InputIterator last, T dt)
    {
      for (InputIterator p = first; p != last; ++p)
      {
        const size_t o = I.offset(*p);
        I[o] += dt * (*this)[o];
      }

      return true;
    }
  };


  //! Midpoint time integrator
  template <typename T, int N>
  class Midpoint : public Image<T, N>
  {
  private:
    Image<T, N>& I;
    Image<T, N> tmp;
    int substep{0};

  public:
    Midpoint(Image<T, N>& _I)
      : Image<T, N>(_I)
      , I{_I}
      , tmp{_I}
    {
    }

    template <typename InputIterator>
    bool step(InputIterator first, InputIterator last, T dt)
    {
      // First substep
      if (substep == 0)
      {
        for (InputIterator p = first; p != last; ++p)
        {
          const int o = I.offset(*p);
          tmp[o] = I[o];
          I[o] += (dt / T(2)) * (*this)[o];
        }

        substep++;
        return false;
      }

      // Second substep
      for (InputIterator p = first; p != last; ++p)
      {
        const int o = I.offset(*p);
        I[o] = tmp[o] + dt * (*this)[o];
      }

      substep = 0;

      return true;
    }
  };

  // @TODO: Runge-Kutta RK4 integrator.
  //!  @}

}  // namespace DO::Sara
