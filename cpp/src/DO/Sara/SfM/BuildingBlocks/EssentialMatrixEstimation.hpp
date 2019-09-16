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

//! @{
//! @brief Essential matrix estimation.
DO_SARA_EXPORT
auto estimate_essential_matrix(
    const std::vector<Match>& Mij,
    const KeypointList<OERegion, float>& ki,
    const KeypointList<OERegion, float>& kj,
    const Eigen::Matrix3d& Ki_inv, const Eigen::Matrix3d& Kj_inv,
    int num_samples, double err_thres)
  -> std::tuple<EssentialMatrix, Tensor_<bool, 1>, Tensor_<int, 1>>;

DO_SARA_EXPORT
auto estimate_essential_matrices(const std::string& dirpath,      //
                                 const std::string& h5_filepath,  //
                                 int num_samples,                 //
                                 double noise,                    //
                                 int min_F_inliers,               //
                                 bool overwrite, bool debug) -> void;

DO_SARA_EXPORT
auto inspect_essential_matrices(const std::string& dirpath,
                                const std::string& h5_filepath,
                                int display_step, bool wait_key) -> void;
//! @}

} /* namespace DO::Sara */
