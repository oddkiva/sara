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

  //! @{
  //! @brief Fundamental matrix estimation.
  DO_SARA_EXPORT
  auto estimate_fundamental_matrix(const std::vector<Match>& Mij,
                                   const KeypointList<OERegion, float>& ki,
                                   const KeypointList<OERegion, float>& kj,
                                   int num_samples, double err_thres)
      -> std::tuple<FundamentalMatrix, Tensor_<bool, 1>, Tensor_<int, 1>>;

  DO_SARA_EXPORT
  auto estimate_fundamental_matrices(const std::string& dirpath,
                                     const std::string& h5_filepath,
                                     bool overwrite, bool debug,
                                     bool wait_key = false) -> void;

  DO_SARA_EXPORT
  auto check_epipolar_constraints(const Image<Rgb8>& Ii, const Image<Rgb8>& Ij,
                                  const FundamentalMatrix& F,
                                  const std::vector<Match>& Mij,
                                  const TensorView_<int, 1>& sample_best,
                                  const TensorView_<bool, 1>& inliers,
                                  int display_step, bool wait_key = true)
      -> void;

  DO_SARA_EXPORT
  auto inspect_fundamental_matrices(const std::string& dirpath,
                                    const std::string& h5_filepath,
                                    int display_step, bool wait_key) -> void;
  //! @}

  //! @}

} /* namespace DO::Sara */
