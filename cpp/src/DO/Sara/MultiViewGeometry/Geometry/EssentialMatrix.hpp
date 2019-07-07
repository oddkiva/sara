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

#include <DO/Sara/MultiViewGeometry/Geometry/FundamentalMatrix.hpp>


namespace DO { namespace Sara {

  class EssentialMatrix : public FundamentalMatrix
  {
    using base_type = FundamentalMatrix;

  public:
    using matrix_type = typename base_type::matrix_type;
    using point_type = typename base_type::point_type;
    using vector_type = point_type;

    using motion_type = std::pair<matrix_type, vector_type>;

    EssentialMatrix() = default;

    EssentialMatrix(const matrix_type& E)
      : base_type{E}
    {
    }

    auto extract_candidate_camera_motions() const
        -> std::array<motion_type, 4>;
  };

} /* namespace Sara */
} /* namespace DO */
