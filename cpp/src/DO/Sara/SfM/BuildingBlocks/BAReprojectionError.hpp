// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2024-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>


namespace DO::Sara {

  struct ReprojectionError
  {
    static constexpr auto ResidualDimension = 2;
    static constexpr auto IntrinsicParameterCount = 4;  // fx, fy, u0, v0
    static constexpr auto ExtrinsicParameterCount = 6;
    static constexpr auto PointDimension = 3;

    ReprojectionError(double observed_x, double observed_y)
      : observed_x{observed_x}
      , observed_y{observed_y}
    {
    }

    template <typename T>
    bool
    operator()(const T* const extrinsics,  // (1) extrinsic camera parameters
               const T* const intrinsics,  // (2) intrinsic camera parameters
               const T* const point,       // (3) 3D point
               T* residuals) const
    {
      T p[3];

      // Rotate the point.
      ceres::AngleAxisRotatePoint(extrinsics, point, p);
      // Translate the point.
      const auto t = extrinsics + 3;
      p[0] += t[0];
      p[1] += t[1];
      p[2] += t[2];

      // Normalized camera coordinates.
      T xp = p[0] / p[2];
      T yp = p[1] / p[2];

      // Apply the internal parameters.
      const auto& fx = intrinsics[0];
      const auto& fy = intrinsics[1];
      const auto& u0 = intrinsics[2];
      const auto& v0 = intrinsics[3];
      const auto predicted_x = fx * xp + u0;
      const auto predicted_y = fy * yp + v0;

      residuals[0] = predicted_x - T(observed_x);
      residuals[1] = predicted_y - T(observed_y);

      return true;
    }

    static ceres::CostFunction* create(const double observed_x,
                                       const double observed_y)
    {
      return new ceres::AutoDiffCostFunction<
          ReprojectionError, ResidualDimension,  //
          ExtrinsicParameterCount, IntrinsicParameterCount, PointDimension>{
          new ReprojectionError{observed_x, observed_y}  //
      };
    }

    double observed_x;
    double observed_y;
  };

}  // namespace DO::Sara
