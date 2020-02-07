// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SHAKTI_MULTIARRAY_STRIDES_HPP
#define DO_SHAKTI_MULTIARRAY_STRIDES_HPP

#include <DO/Shakti/MultiArray/Matrix.hpp>


namespace DO { namespace Shakti {

  struct RowMajorStrides
  {
    template <typename Index, int N>
    __host__ __device__
    static inline Vector<Index, N> compute(const Vector<Index, N>& sizes)
    {
      Vector<Index, N> strides;
      strides(N - 1) = 1;
      for (int i = N - 2; i >= 0; --i)
        strides(i) = strides(i + 1) * sizes(i + 1);
      return strides;
    }
  };

  struct ColumnMajorStrides
  {
    template <typename Index, int N>
    __host__ __device__
    static inline Vector<Index, N> compute(const Vector<Index, N>& sizes)
    {
      Vector<Index, N> strides;
      strides(0) = 1;
      for (int i = 1; i < N; ++i)
        strides(i) = strides(i-1) * sizes(i-1);
      return strides;
    }
  };

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_MULTIARRAY_STRIDES_HPP */