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

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Core/Math/JenkinsTraub.hpp>
#include <DO/Sara/Core/Math/NewtonRaphson.hpp>
#include <DO/Sara/Core/Pixel/PixelTraits.hpp>

#include <DO/Sara/MultiViewGeometry/Camera/PinholeCamera.hpp>


namespace DO::Sara {

  template <typename T>
  struct RadialRationalCorrectionFunction
  {
    Eigen::Matrix<T, 2, 1> k;
    Eigen::Matrix<T, 2, 1> distortion_center;

    auto operator()(const Eigen::Matrix<T, 2, 1>& xd) const
        -> Eigen::Matrix<T, 2, 1>
    {
      const auto r2 = (xd - distortion_center).squaredNorm();
      const auto r4 = r2 * r2;
      const auto distortion_factor = 1 / (1 + k(0) * r2 + k(1) * r4);
      const auto& c = distortion_center;
      return c + distortion_factor * (xd - c);
    }

    auto inverse(const Eigen::Matrix<T, 2, 1>& xu) const
        -> Eigen::Matrix<T, 2, 1>
    {
      const auto& c = distortion_center;
      const auto ru = (xu - c).squaredNorm();

      auto P = Univariate::UnivariatePolynomial<T>{4};
      P[0] = 1;
      P[1] = -1;
      P[2] = ru * k(0);
      P[3] = 0;
      P[4] = ru * k(1);

      auto solver = Univariate::NewtonRaphson<T>{P};
      const auto rd = solver(ru);

      return c + rd / ru * (xu - c);
    }
  };

  template <typename T>
  struct RadialPolynomialCorrectionFunction
  {
    Eigen::Matrix<T, 2, 1> k;
    Eigen::Matrix<T, 2, 1> distortion_center;

    auto operator()(const Eigen::Matrix<T, 2, 1>& xd) const
        -> Eigen::Matrix<T, 2, 1>
    {
      const auto r2 = (xd - distortion_center).squaredNorm();
      const auto r4 = r2 * r2;
      const auto distortion_factor = 1 + k(0) * r2 + k(1) * r4;
      const auto& c = distortion_center;
      return c + distortion_factor * (xd - c);
    }

    auto inverse(const Eigen::Matrix<T, 2, 1>& xu, int max_iter = 20,
                 T eps = T{1e-6}) const -> Eigen::Matrix<T, 2, 1>
    {
      const auto& c = distortion_center;
      const auto ru = (xu - c).squaredNorm();

      auto corrected_radius = [this](T rd) {
        const auto rd2 = rd * rd;
        const auto rd3 = rd2 * rd;
        const auto rd5 = rd2 * rd3;
        return rd + k(0) * rd3 + k(1) * rd5;
      };

      auto rd_min = T{};
      auto rd_max = 2 * ru;
      // Find a smaller upper bound by exploiting monotonicity.
      auto corrected_rd = corrected_radius(rd_max);
      while (corrected_radius(rd_max) < ru)
      {
        rd_min = rd_max;
        rd_max = 2 * rd_max;
      }

      // Apply bisection.
      auto rd_estimate = static_cast<T>(0.5) * (rd_min + rd_max);
      auto ru_at_rd_estimate = corrected_radius(rd_estimate);
      auto iter = 0;
      while (iter < max_iter &&
             std::abs(ru_at_rd_estimate - ru) > eps)
      {
        if (ru_at_rd_estimate < ru)
          rd_min = ru_at_rd_estimate;
        else
          rd_max = ru_at_rd_estimate;

        rd_estimate = T{0.5} * (rd_min + rd_max);

        ++iter;
      }

      return c + rd_estimate / ru * (xu - c);
    }
  };

  template <typename T, typename CorrectionFunction>
  struct CameraCorrectionModel : PinholeCamera<T>
  {
    static constexpr auto eps = static_cast<T>(1e-8);

    using correction_function_type = CorrectionFunction;

    //! @brief Types.
    using base_type = PinholeCamera<T>;
    using vector2_type = typename base_type::vector2_type;
    using vector3_type = typename base_type::vector3_type;
    using matrix2_type = Eigen::Matrix<T, 2, 2>;
    using matrix3_type = typename base_type::matrix3_type;

    using base_type::image_sizes;
    using base_type::K;
    using base_type::K_inverse;

    // Correction model.
    correction_function_type correction_function;

    inline auto undistort(const vector2_type& xd) const -> vector2_type
    {
      return correction_function.apply(xd);
    }

    inline auto distort(const vector2_type& xu) const -> vector2_type
    {
      return correction_function.apply_inverse(xu);
    }

    inline auto project(const vector3_type& x) const -> vector2_type
    {
      const Eigen::Vector2f pixel_coords = (K * x).hnormalized();
      return correction_function.apply_inverse(pixel_coords);
    }

    inline auto backproject(const vector2_type& x) const -> vector3_type
    {
      const auto xu = correction_function.apply(x);
      return K_inverse * xu.homogeneous();
    }
  };

}  // namespace DO::Sara
