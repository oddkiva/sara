// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2024 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include "DO/Sara/MultiViewGeometry/PointCorrespondenceList.hpp"
#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  template <typename T>
  using PointRayCorrespondenceList = PointCorrespondenceList<T>;

  template <typename T>
  using PointRayCorrespondenceSubsetList = PointCorrespondenceSubsetList<T>;

}  // namespace DO::Sara
