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

#pragma once

#include <DO/Sara/Core/MultiArray/MultiArray.hpp>
#include <DO/Sara/Core/MultiArray/MultiArrayView.hpp>
#include <DO/Sara/Core/Numpy.hpp>


namespace DO::Sara {

  /*!
   *  @addtogroup MultiArray
   *  @{
   */

  //! @brief Axis slice descriptor structure.
  struct AxisSlice
  {
    int start{0};
    int stop{0};
    int step{1};
  };


  //! @brief Sliced ND-view structure.
  template <typename T, int N, int O>
  struct ViewSliced
  {
    using vector_type = Matrix<int, N, 1>;
    using view_type = MultiArrayView<T, N, O>;

    auto operator()(const vector_type& x) -> T&
    {
      if ((x.array() < 0).any() || ((slice_sizes - x).array() <= 0).any())
        throw std::runtime_error{
            "Coordinates are not in the valid slices range!"};
      const vector_type y = start + x.dot(steps);
      return view(y);
    }

    auto operator()(const Matrix<int, N, 1>& x) const -> const T&
    {
      if ((x.array() < 0).any() || ((slice_sizes - x).array() <= 0).any())
        throw std::runtime_error{
            "Coordinates are not in the valid slices range!"};
      const vector_type y = start + x.dot(steps);
      return view(y);
    }

    auto begin() const
    {
      return view.begin_stepped_subarray(start, stop, steps);
    }

    auto end() const
    {
      return view.end_stepped_subarray(start, stop, steps);
    }

    auto sizes() const -> const vector_type&
    {
      return slice_sizes();
    }

    auto make_copy() const
    {
      auto view_copy = MultiArray<T, N, O>{slice_sizes};
      std::transform(std::begin(*this), std::end(*this), std::begin(view_copy),
                     [](const auto& v) { return v; });
      return view_copy;
    }

    view_type view;
    vector_type start;
    vector_type stop;
    vector_type steps;
    vector_type slice_sizes;
  };


  template <typename T, int N, int O>
  auto slice(const MultiArrayView<T, N, O>& x,
             const std::vector<AxisSlice>& slices)
  {
    const auto ixs = range(N);
    auto start = Matrix<int, N, 1>{};
    auto stop = Matrix<int, N, 1>{};
    auto step = Matrix<int, N, 1>{};
    std::for_each(std::begin(ixs), std::end(ixs), [&](int i) {
      start[i] = slices[i].start;
      stop[i] = slices[i].stop;
      step[i] = slices[i].step;
    });

    const auto slice_sizes =
        x.begin_stepped_subarray(start, stop, step).stepped_subarray_sizes();

    return ViewSliced<T, N, O>{x, start, stop, step, slice_sizes};
  }

  //! @}

} /* namespace DO::Sara */
