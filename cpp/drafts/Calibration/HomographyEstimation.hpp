// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <drafts/Calibration/Chessboard.hpp>


//! @brief Estimate the homography using the Direct Linear Transform method.
auto estimate_H(const Eigen::MatrixXd& p1, const Eigen::MatrixXd& p2)
    -> Eigen::Matrix3d;

auto estimate_H(const DO::Sara::ChessboardCorners& chessboard)
    -> Eigen::Matrix3d;
