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

#pragma once

#include <DO/Sara/ImageProcessing.hpp>


namespace DO::Sara {

  template <typename T>
  struct Corner
  {
    Eigen::Vector2<T> coords;
    float score;
    float scale;

    inline auto position() const -> const Eigen::Vector2<T>&
    {
      return coords;
    }

    inline auto operator<(const Corner& other) const -> bool
    {
      return score < other.score;
    }
  };

  // Select the local maxima of the cornerness functions.
  auto select(const ImageView<float>& cornerness,
              const float sigma_D, const float sigma_I,
              const float cornerness_adaptive_thres, const int border)
      -> std::vector<Corner<int>>;

}  // namespace DO::Sara
