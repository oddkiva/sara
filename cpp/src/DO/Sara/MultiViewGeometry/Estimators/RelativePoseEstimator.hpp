
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

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/MultiViewGeometry/Estimators/EssentialMatrixEstimators.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/TwoViewGeometry.hpp>

namespace DO::Sara {

enum class CheiralityCriterion : std::uint8_t
{
  CHEIRAL_COMPLETE,
  CHEIRAL_MOST,
  NONE
};


template <typename Method>
struct RelativePoseEstimator : Method
{
  static constexpr auto N = Method::num_points;

  CheiralityCriterion cheiral_criterion{CheiralityCriterion::NONE};

  RelativePoseEstimator(CheiralityCriterion c = CheiralityCriterion::NONE)
    : cheiral_criterion{c}
  {
  }

  auto operator()(const Matrix<double, 3, N>& left,
                  const Matrix<double, 3, N>& right) const
      -> std::vector<TwoViewGeometry>;
};

#ifndef DO_SARA_EXPORTS
extern template struct RelativePoseEstimator<NisterFivePointAlgorithm>;
extern template struct RelativePoseEstimator<SteweniusFivePointAlgorithm>;
#endif

} /* namespace DO::Sara */
