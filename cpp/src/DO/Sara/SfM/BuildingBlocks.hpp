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

#include <DO/Sara/MultiViewGeometry.hpp>

#include <string>


namespace DO::Sara {

//! @{
//! @brief Keypoint detection.
auto detect_keypoints(const std::string& dirpath,
                      const std::string& h5_filepath, bool overwrite) -> void;

auto read_keypoints(const std::string& dirpath, const std::string& h5_filepath)
    -> void;
//! @}


//! @{
//! @brief Keypoint matching.
auto match(const KeypointList<OERegion, float>& keys1,
           const KeypointList<OERegion, float>& keys2,
           float lowe_ratio = 0.6f)
    -> std::vector<Match>;

auto match_keypoints(const std::string& dirpath, const std::string& h5_filepath,
                     bool overwrite) -> void;
//! @}


//! @{
//! @brief Fundamental matrix estimation.
auto estimate_fundamental_matrix(const std::vector<Match>& Mij,
                                 const KeypointList<OERegion, float>& ki,
                                 const KeypointList<OERegion, float>& kj,
                                 int num_samples, double err_thres)
    -> std::tuple<FundamentalMatrix, Tensor_<bool, 1>, Tensor_<int, 1>>;

auto estimate_fundamental_matrices(const std::string& dirpath,
                                   const std::string& h5_filepath,
                                   bool overwrite, bool debug) -> void;

auto check_epipolar_constraints(const Image<Rgb8>& Ii,
                                const Image<Rgb8>& Ij,
                                const FundamentalMatrix& F,
                                const std::vector<Match>& Mij,
                                const TensorView_<int, 1>& sample_best,
                                const TensorView_<bool, 1>& inliers,
                                int display_step,
                                bool wait_key = true) -> void;

auto inspect_fundamental_matrices(const std::string& dirpath,
                                  const std::string& h5_filepath,
                                  int display_step,
                                  bool wait_key) -> void;
//! @}


//! @{
//! @brief Essential matrix estimation.
auto estimate_essential_matrix(
    const std::vector<Match>& Mij,
    const KeypointList<OERegion, float>& ki,
    const KeypointList<OERegion, float>& kj,
    const Eigen::Matrix3d& Ki_inv, const Eigen::Matrix3d& Kj_inv,
    int num_samples, double err_thres)
  -> std::tuple<EssentialMatrix, Tensor_<bool, 1>, Tensor_<int, 1>>;

auto estimate_essential_matrices(const std::string& dirpath,      //
                                 const std::string& h5_filepath,  //
                                 int num_samples,                 //
                                 double noise,                    //
                                 int min_F_inliers,               //
                                 bool overwrite, bool debug) -> void;

auto inspect_essential_matrices(const std::string& dirpath,
                                const std::string& h5_filepath,
                                int display_step, bool wait_key) -> void;
//! @}

} /* namespace DO::Sara */
