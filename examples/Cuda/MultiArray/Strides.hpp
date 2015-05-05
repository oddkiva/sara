#pragma once

#include "Matrix.hpp"

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

} /* namespace Dhara */
} /* namespace DO */