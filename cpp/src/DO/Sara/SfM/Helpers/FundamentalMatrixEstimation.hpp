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

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/MultiViewGeometry.hpp>


namespace DO::Sara {

  //! @addtogroup SfM
  //! @{

  //! @brief Helper to estimate the fundamental matrix.
  auto estimate_fundamental_matrix(const std::vector<Match>& Mij,
                                   const KeypointList<OERegion, float>& ki,
                                   const KeypointList<OERegion, float>& kj,
                                   const int num_samples,
                                   const double err_thres)
      -> std::tuple<FundamentalMatrix, Tensor_<bool, 1>, Tensor_<int, 1>>;

  //! @brief Inspect visually the epipolar constraints.
  auto check_epipolar_constraints(const Image<Rgb8>& Ii, const Image<Rgb8>& Ij,
                                  const FundamentalMatrix& F,
                                  const std::vector<Match>& Mij,
                                  const TensorView_<int, 1>& sample_best,
                                  const TensorView_<bool, 1>& inliers,
                                  int display_step, bool wait_key = true)
      -> void;

  //! @}

} /* namespace DO::Sara */
