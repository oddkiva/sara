// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/MultiArray/Padding.hpp>


namespace DO { namespace Sara {

  template <typename DF, typename F>
  class NeumannPadding
  {
  public:
    inline NeumannPadding() = default;

    template <int N, int O>
    inline auto at(MultiArrayView<F, N, O>& f, const Matrix<int, N, 1>& x) const
        -> const F&
    {
      if (x.minCoeff() < 0)
        return f(x) + _df_x * x;

      if ((x - f.sizes()).minCoeff() >= 0)
        f(x) + _df_x*(x - f.sizes());

      return f(x);
    }

  private:
    DF _df_x;
  };

} /* namespace Sara */
} /* namespace DO */
