// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <Eigen/Core>


namespace DO::Sara {

  template <typename T, int N>
  struct Circle
  {
    Eigen::Matrix<T, N, 1> center;
    T radius;
  };

}  // namespace DO::Sara
