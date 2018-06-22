// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <iterator>
#include <sstream>
#include <stdexcept>

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Meta.hpp>


namespace DO { namespace Sara {

  //! @ingroup MultiArrayIterators
  //! @{

  //! @brief The offset computer class for N-dimensional arrays.
  template <typename Index, int N>
  inline Index jump(const Matrix<Index, N, 1>& offset,
                    const Matrix<Index, N, 1>& strides)
  {
    static_assert(
      std::numeric_limits<Index>::is_integer,
      "Index must be integral");
    return offset.dot(strides);
  }

  //! @brief Computes the strides from the sizes of an N-dimensional array.
  //! StorageOrder must be either 'Eigen::RowMajor' or 'Eigen::ColMajor'.
  template <int StorageOrder> struct StrideComputer;

  //! @brief Increment the current position in an N-dimensional array.
  //! StorageOrder must be either 'Eigen::RowMajor' or 'Eigen::ColMajor'.
  template <int StorageOrder> struct PositionIncrementer;

  //! @brief Decrement the current position in an N-dimensional array.
  //! StorageOrder must be either 'Eigen::RowMajor' or 'Eigen::ColMajor'.
  template <int StorageOrder> struct PositionDecrementer;

  //! @brief Computes the strides from the sizes of an N-dimensional array.
  //! Specialized class for 'Eigen::RowMajor'.
  template <> struct StrideComputer<RowMajor>
  {
    template <typename Index, int N>
    static Matrix<Index, N, 1> eval(const Matrix<Index, N, 1>& sizes)
    {
      static_assert(N > 0, "N must be positive");
      static_assert(
        std::numeric_limits<Index>::is_integer,
        "Index must be integral");
      Matrix<Index, N, 1> strides;
      strides[N-1] = 1;
      for (int i = N-2; i >= 0; --i)
        strides[i] = strides[i+1]*sizes[i+1];
      return strides;
    }
  };

  //! @brief Computes the strides from the sizes of an N-dimensional array.
  //! Specialized class for 'Eigen::ColMajor'.
  template <> struct StrideComputer<ColMajor>
  {
    template <typename Index, int N>
    static Matrix<Index, N, 1> eval(const Matrix<Index, N, 1>& sizes)
    {
      static_assert(N > 0, "N must be positive");
      static_assert(
        std::numeric_limits<Index>::is_integer,
        "Index must be integral");
      Matrix<Index, N, 1> strides;
      strides[0] = 1;
      for (int i = 1; i < N; ++i)
        strides[i] = strides[i-1]*sizes[i-1];
      return strides;
    }
  };

  //! @brief Increment the current position in an N-dimensional array.
  //! StorageOrder must be either 'Eigen::RowMajor'.
  template <> struct PositionIncrementer<RowMajor>
  {
    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& start,
                             const Matrix<Index, N, 1>& end)
    {
      for (int i = N - 1; i >= 0; --i)
      {
        ++coords[i];
        if (coords[i] != end[i])
          return;
        coords[i] = start[i];
      }
      if (coords[0] == start[0])
        stop = true;
    }

    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& start,
                             const Matrix<Index, N, 1>& end,
                             const Matrix<Index, N, 1>& steps)
    {
      for (int i = N - 1; i >= 0; --i)
      {
        coords[i] += steps[i];
        if (coords[i] < end[i])
          return;
        coords[i] = start[i];
      }

      if (coords[0] == start[0])
        stop = true;
    }

    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& sizes)
    {
      apply<Index, N>(coords, stop, Matrix<Index, N, 1>::Zero(), sizes);
    }
  };

  //! @brief Increment the current position in an N-dimensional array.
  //! StorageOrder must be either 'Eigen::ColMajor'.
  template <> struct PositionIncrementer<ColMajor>
  {
    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& start,
                             const Matrix<Index, N, 1>& end)
    {
      for (int i = 0; i < N; ++i)
      {
        ++coords[i];
        if (coords[i] != end[i])
          return;
        coords[i] = start[i];
      }
      if (coords[N - 1] == start[N - 1])
        stop = true;
    }

    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& start,
                             const Matrix<Index, N, 1>& end,
                             const Matrix<Index, N, 1>& steps)
    {
      for (int i = 0; i < N; ++i)
      {
        coords[i] += steps[i];
        if (coords[i] < end[i])
          return;
        coords[i] = start[i];
      }
      if (coords[N - 1] >= start[N - 1])
        stop = true;
    }

    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& sizes)
    {
      apply<Index, N>(coords, stop, Matrix<Index, N, 1>::Zero(), sizes);
    }
  };

  //! @brief Decrement the current position in an N-dimensional array.
  //! StorageOrder must be either 'Eigen::RowMajor'.
  template <> struct PositionDecrementer<RowMajor>
  {
    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& start,
                             const Matrix<Index, N, 1>& end)
    {
      for (int i = N - 1; i >= 0; --i)
      {
        if (coords[i] != start[i])
        {
          --coords[i];
          return;
        }

        coords[i] = end[i] - 1;
      }

      if (coords[0] == end[0] - 1)
        stop = true;
    }

    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& start,
                             const Matrix<Index, N, 1>& end,
                             const Matrix<Index, N, 1>& steps)
    {
      for (int i = N - 1; i >= 0; --i)
      {
        if (coords[i] >= start[i] + steps[i])
        {
          coords[i] -= steps[i];
          return;
        }

        const auto n_i = (end[i] - start[i]) / steps[i];
        coords[i] = start[i] + n_i * steps[i];
      }

      const auto n_0 = (end[0] - start[0]) / steps[0];
      if (coords[0] == start[0] + n_0 * steps[0])
        stop = true;
    }

    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& sizes)
    {
      apply<Index, N>(coords, stop, Matrix<Index, N, 1>::Zero(), sizes);
    }
  };

  //! @brief Decrement the current position in an N-dimensional array.
  //! StorageOrder must be either 'Eigen::ColMajor'.
  template <> struct PositionDecrementer<ColMajor>
  {
    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& start,
                             const Matrix<Index, N, 1>& end)
    {
      for (int i = 0; i < N; ++i)
      {
        if (coords[i] != start[i])
        {
          --coords[i];
          return;
        }

        coords[i] = end[i] - 1;
      }

      if (coords[N - 1] == end[N - 1] - 1)
        stop = true;
    }

    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& start,
                             const Matrix<Index, N, 1>& end,
                             const Matrix<Index, N, 1>& steps)
    {
      for (int i = 0; i < N; ++i)
      {
        if (coords[i] >= start[i] + steps[i])
        {
          coords[i] -= steps[i];
          return;
        }

        const auto n_i = (end[i] - start[i]) / steps[i];
        coords[i] = start[i] + n_i * steps[i];
      }

      const auto n_last = (end[N - 1] - start[N - 1]) / steps[N - 1];
      if (coords[N - 1] == start[N - 1] + n_last * steps[N - 1])
        stop = true;
    }

    template <typename Index, int N>
    static inline void apply(Matrix<Index, N, 1>& coords, bool& stop,
                             const Matrix<Index, N, 1>& sizes)
    {
      apply<Index, N>(coords, stop, Matrix<Index, N, 1>::Zero(), sizes);
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
