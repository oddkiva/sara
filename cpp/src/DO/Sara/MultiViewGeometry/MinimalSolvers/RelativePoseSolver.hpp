
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

#include <Eigen/StdVector>

#include <DO/Sara/Core/Numpy.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/TwoViewGeometry.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/EssentialMatrixSolvers.hpp>

#include <iterator>


namespace DO::Sara {

  //! @addtogroup MinimalSolvers
  //! @{

  //! @brief Relative pose estimator.
  template <typename Method>
  struct RelativePoseSolver : Method
  {
    using model_type = TwoViewGeometry;
    using data_point_type = typename Method::data_point_type;

    static constexpr auto N = Method::num_points;
    static constexpr auto num_points = Method::num_points;
    static constexpr auto num_models = Method::num_models * 4;

    RelativePoseSolver() = default;

    auto operator()(const Matrix<double, 3, N>& left,
                    const Matrix<double, 3, N>& right) const
        -> std::vector<TwoViewGeometry>
    {
      const auto Es = this->Method::find_essential_matrices(left, right);
      // There is no two-view geometry if we could not compute any essential
      // matrices from the minimal subset.
      if (Es.empty())
        return {};

      // There is 4 possible motions for each essential matrix.
      auto indices = range(static_cast<int>(Es.size()));
      auto motions = Tensor_<Motion, 2>{{static_cast<int>(Es.size()), 4}};
      std::for_each(std::begin(indices), std::end(indices), [&](int i) {
        motions[i] = tensor_view(extract_relative_motion_horn(Es[i]));
      });

      // Retrieve the 3D point coordinates by triangulation.
      //
      // The 3D point coordinates are expressed in the first camera frame.
      auto geometries = std::vector<TwoViewGeometry>(motions.size());
      std::transform(
          std::begin(motions), std::end(motions), std::begin(geometries),
          [&left, &right](const auto& m) { return two_view_geometry(m, left, right); });

      // Remove geometries where all the points violates the cheirality.
      static_assert(N == 5, "The implementation is valid for 5-point solver only!");
      const auto last = std::remove_if(std::begin(geometries),
                                      std::end(geometries),
                                      [&](const auto& g) {
#if defined(I_AM_SURE_THERE_IS_NO_POINT_AT_INFINITY)
                                        return g.cheirality.count() != N;
#else
                                        return g.cheirality.count() < N - 1;
#endif
                                      });
      geometries.erase(last, geometries.end());
      return geometries;
    }

    auto operator()(const data_point_type& X) const
        -> std::vector<TwoViewGeometry>
    {
      const auto left = X[0].colmajor_view().matrix();
      const auto right = X[1].colmajor_view().matrix();
      return this->operator()(left, right);
    }
  };

  extern template struct RelativePoseSolver<NisterFivePointAlgorithm>;
  extern template struct RelativePoseSolver<SteweniusFivePointAlgorithm>;

  //! @}

} /* namespace DO::Sara */
