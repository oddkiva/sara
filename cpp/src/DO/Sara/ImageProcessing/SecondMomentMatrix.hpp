// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/Image/Image.hpp>
#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  /*!
   * @ingroup Differential
   * @{
   */

  /*!
   * @brief Second-moment matrix helper class in order to use
   * DO::Image<T>::compute<SecondMomentMatrix>()
   */
  struct SecondMomentMatrix
  {
    template <typename GradientField>
    using Scalar = typename GradientField::pixel_type::Scalar;

    template <typename GradientField>
    using Matrix =
        Eigen::Matrix<Scalar<GradientField>, GradientField::Dimension,
                      GradientField::Dimension>;

    template <typename GradientField>
    using MatrixField = Image<Matrix<GradientField>, GradientField::Dimension>;

    template <typename GradientField>
    auto operator()(const GradientField& in) -> MatrixField<GradientField>
    {
      auto out = MatrixField<GradientField>{in.sizes()};

      auto out_i = out.begin();
      auto in_i = in.begin();
      for (; in_i != in.end(); ++in_i, ++out_i)
        *out_i = *in_i * in_i->transpose();

      return out;
    }
  };

  //! @brief Calculate the second moment matrix
  //! Optimized with Halide.
  auto second_moment_matrix(const ImageView<float>& fx,
                            const ImageView<float>& fy,  //
                            ImageView<float>& mxx,       //
                            ImageView<float>& myy,       //
                            ImageView<float>& mxy) -> void;

  inline auto second_moment_matrix(const TensorView_<float, 3>& f)
      -> Tensor_<float, 3>
  {
    auto m = Tensor_<float, 3>{3, f.size(1), f.size(2)};
    const auto fx = image_view(f[0]);
    const auto fy = image_view(f[1]);
    auto mxx = image_view(m[0]);
    auto myy = image_view(m[1]);
    auto mxy = image_view(m[2]);
    second_moment_matrix(fx, fy, mxx, myy, mxy);
    return m;
  }
  //! @}

}  // namespace DO::Sara
