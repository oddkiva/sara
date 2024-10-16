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

#include <DO/Sara/MultiViewGeometry/Camera/PinholeCamera.hpp>


namespace DO::Sara {

  template <typename T>
  struct DrapLefevreRadialModel : PinholeCamera<T>
  {
    using base_type = PinholeCamera<T>;
    using scalar_type = T;
    using vector2_type = typename base_type::vector2_type;
    using vector3_type = typename base_type::vector3_type;
    using vector4_type = Eigen::Matrix<T, 4, 1>;

    using base_type::image_sizes;
    using base_type::K;
    using base_type::K_inverse;

    //! @brief Radial distortion coefficients.
    struct DistortionCoefficients
    {
      vector3_type k;
    };

    struct CorrectionCoefficients
    {
      vector4_type k;
    };

    //! @brief Cached variable for the inverse distortion calculation.
    //! cf. publication:
    //!   An Exact Formula for Calculating Inverse Radial Lens Distortions,
    //!   Drap and Lefevre.
    inline auto invert_radial_function(const Eigen::Ref<vector3_type>& k) const -> vector4_type
    {
      auto k_inverse = vector4_type{};

      k_inverse(0) = -k(0);
      k_inverse(1) = 3 * std::pow(k(0), 2) - k(1);
      k_inverse(2) = 8 * k(0) * k(1) - 12 * std::pow(k(1), 3) - k(2);

      const auto k3 = 0;
      k_inverse(3) =   55 * std::pow(k(0), 4)         //
                     - 55 * std::pow(k(0), 2) * k(1)  //
                     + 5 * std::pow(k(1), 2)          //
                     + 10 * k(0) * k(2)               //
                     - k3;
      k_inverse(4) = -273 * std::pow(k(0), 5) //
                     +364 * std::pow(k(0), 3) * k(1) //
                     -78 * k(0) * std::pow(k(1), 2) //
                     -78 * std::pow(k(0), 2) * k(2) //
                     +12 * k(1) * k(2) //
                     +12 * k(0) * k3;
    }
  };

}  // namespace DO::Sara
