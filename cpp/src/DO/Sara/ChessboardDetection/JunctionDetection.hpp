// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>


namespace DO::Sara {

  //! @brief Kept for historical reasons although Harris's corner detection
  //! handles the computational efficiency better.
  //! @{
  template <typename T>
  struct Junction
  {
    Eigen::Matrix<T, 2, 1> p;
    float score;

    inline auto position() const -> const Eigen::Matrix<T, 2, 1>&
    {
      return p;
    }

    inline auto operator<(const Junction& other) const
    {
      return score < other.score;
    }
  };

  auto junction_map(const ImageView<float>& image,
                    const ImageView<Eigen::Vector2f>& gradients,
                    const float sigma) -> Image<float>;

  auto extract_junctions(const ImageView<float>& junction_map, const int radius)
      -> std::vector<Junction<int>>;
  //! @}

}  // namespace DO::Sara
