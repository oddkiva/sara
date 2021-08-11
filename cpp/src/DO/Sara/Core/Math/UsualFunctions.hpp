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


namespace DO::Sara {

  //! @{
  //! @brief Some functions to avoid using std::pow due to different compiler
  //! implementations.
  //!
  //! They are also faster than std::pow(x, N) in N = 2, 3, 4.
  template <typename T>
  inline constexpr auto square(T x) noexcept -> T
  {
    return x * x;
  }

  template <typename T>
  inline constexpr auto cubic(T x) noexcept -> T
  {
    return x * x * x;
  }

  template <typename T>
  inline constexpr auto quartic(T x) noexcept -> T
  {
    return square(square(x));
  }
  //! @}
}
