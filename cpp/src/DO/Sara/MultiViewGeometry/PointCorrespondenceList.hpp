// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2024 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  //! @brief The data structure to store the list of point correspondences.
  //!
  //! A point correspondence is denoted as $(x[i], y[i])$.
  //!
  //! In my opinion, 'x' and 'y' are the lightest and most agnostic notations I
  //! could think of. We do lose expressivity because they may be too neutral,
  //! but it still feels natural and mathematical at the same time.
  //!
  //! Notice that we are not assuming anything about the dimensions of x[i] and
  //! y[i].
  //! - x[i] and y[i] don't have to have the same dimensions.
  //! - (x[i], y[i]) can be 2D point <-> 2D point correspondence in two
  //!   different images, or
  //! - (x[i], y[i]) can be a 3D point <-> 2D point correspondence, or
  //! - (x[i], y[i]) can be a 3D scene point <-> 3D ray correspondence.
  template <typename T>
  struct PointCorrespondenceList
  {
    using value_type = std::array<const TensorView_<T, 1>, 2>;

    PointCorrespondenceList() = default;

    PointCorrespondenceList(const TensorView_<int, 2>& M,
                            const TensorView_<T, 2>& x_all,
                            const TensorView_<T, 2>& y_all)
      : x{M.size(0), x_all.size(1)}
      , y{M.size(0), y_all.size(1)}
    {
      auto x_all_mat = x_all.matrix();
      auto y_all_mat = y_all.matrix();
      auto x_matched = x.matrix();
      auto y_matched = y.matrix();
      for (auto m = 0; m < M.size(0); ++m)
      {
        const auto& x_idx = M(m, 0);
        const auto& y_idx = M(m, 1);

        x_matched.row(m) = x_all_mat.row(x_idx);
        y_matched.row(m) = y_all_mat.row(y_idx);
      }
    }

    auto size() const -> int
    {
      return x.size(0);
    }

    auto operator[](const int n) const -> value_type
    {
      return {x[n], y[n]};
    }

    //! @brief The correspondences are: (x[i], y[i]).
    //! @{
    Tensor_<T, 2> x;
    Tensor_<T, 2> y;
    //! @}
  };

  template <typename T>
  struct PointCorrespondenceSubsetList
  {
    using value_type = std::array<TensorView_<T, 2>, 2>;

    auto operator[](const int n) const -> value_type
    {
      return {x[n], y[n]};
    }

    Tensor_<T, 3> x;
    Tensor_<T, 3> y;
  };

}  // namespace DO::Sara
