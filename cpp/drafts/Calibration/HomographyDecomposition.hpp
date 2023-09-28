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

#include <Eigen/Dense>


auto decompose_H_RQ_factorization(const Eigen::Matrix3d& H,
                                  const Eigen::Matrix3d& K,
                                  std::vector<Eigen::Matrix3d>& Rs,
                                  std::vector<Eigen::Vector3d>& ts,
                                  std::vector<Eigen::Vector3d>& ns) -> void;

auto decompose_H_faugeras(const Eigen::Matrix3d& H, const Eigen::Matrix3d& K,
                          std::vector<Eigen::Matrix3d>& Rs,
                          std::vector<Eigen::Vector3d>& ts,
                          std::vector<Eigen::Vector3d>& ns) -> void;
