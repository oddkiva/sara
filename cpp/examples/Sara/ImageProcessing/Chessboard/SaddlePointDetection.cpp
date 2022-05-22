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

#include "SaddlePointDetection.hpp"


namespace DO::Sara {

  auto extract_saddle_points(const ImageView<float>& det_of_hessian,
                             const ImageView<Eigen::Matrix2f>& hessian,
                             float thres) -> std::vector<SaddlePoint>
  {
    auto saddle_points = std::vector<SaddlePoint>{};
    saddle_points.reserve(std::max(hessian.width(), hessian.height()));

    for (auto y = 0; y < hessian.height(); ++y)
    {
      for (auto x = 0; x < hessian.width(); ++x)
      {
        if (det_of_hessian(x, y) > thres)
          continue;

        // // One simple filter...
        // auto qr_factorizer = hessian(x, y).householderQr();
        // const Eigen::Matrix2f R =
        //     qr_factorizer.matrixQR().triangularView<Eigen::Upper>();

        // if (std::abs((R(0, 0) - R(1, 1)) / R(0, 0)) > 0.1f)
        //   continue;

        // TODO: corner filtering by counting the number of zero crossing over a
        // small cicle.

        saddle_points.push_back({
            {x, y}, hessian(x, y), std::abs(det_of_hessian(x, y))  //
        });
      }
    }

    return saddle_points;
  }

}  // namespace DO::Sara
