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

#include <drafts/Halide/MyHalide.hpp>
#include <drafts/Halide/Components/TinyLinearAlgebra.hpp>


namespace DO::Shakti::HalideBackend {

  struct SIFT
  {
    // We construct a grid of NxN histogram of gradients.
    const std::int32_t N{4};
    // Each histogram has 8 bins.
    const std::int32_t O{8};

    const float bin_length_in_scale_unit{3.f};
    float max_bin_value{0.2f};

    // Each image has a resolution scale (in pixels) associated to it.
    float first_scale_of_octave;
    // The resolution scale of the image is not bigger than twice the initial
    // scale of the octave. Otherwise we calculate an equivalent SIFT descriptor
    // on the downsampled image.
    static constexpr float max_scale_factor_of_octave{2.f};
    static constexpr auto two_pi{static_cast<float>(2 * M_PI)};

    auto patch_radius_bound() const
    {
      return static_cast<int>(std::round(first_scale_of_octave *
                                         max_scale_factor_of_octave * (N + 1) /
                                         2.f * std::sqrt(2.f)));
    }

    auto patch_radius(Halide::Expr sigma) const
    {
      return Halide::cast<int>(Halide::round(bin_length_in_scale_unit * sigma *
                                             (N + 1) / 2.f * std::sqrt(2.f)));
    }

    auto compute(const Halide::Func& grad_mag_fn,  //
                 const Halide::Func& grad_ori_fn,  //
                 const Halide::Var& i,             //
                 const Halide::Var& j,             //
                 const Halide::Var& o,             //
                 const Halide::Expr& x,            //
                 const Halide::Expr& y,            //
                 const Halide::Expr& sigma,        //
                 const Halide::Expr& theta)        //
    {
      // Calculate the radius upper-bound.
      const auto r_max = patch_radius_bound();
      // Calculate the radius of the actual patch.
      const auto r_actual = patch_radius(sigma);

      // The reduction domain is actually:
      auto r = Halide::RDom(-r_max, 2 * r_max, -r_max, 2 * r_max);
      r.where(Halide::abs(r.x) < r_actual &&  //
              Halide::abs(r.y) < r_actual);

      // The gradient magnitude is:
      const auto xi = Halide::round(x);
      const auto yi = Halide::round(y);
      const auto mag = grad_mag_fn(xi + r.x, yi + r.y);

      // The orientation in the reoriented patch is:
      const auto ori_shifted = grad_ori_fn(xi + r.x, yi + r.y) - theta;
      // We have:
      // -pi < ori < pi
      // -pi < -theta < pi
      // -2*pi < ori - theta < 2 * pi
      // So we make sure that the orientation is [0, 2*pi[
      const auto ori =
          Halide::select(ori_shifted < 0, ori_shifted + two_pi, ori_shifted);

      auto ori_normalized = Halide::select(ori < 0, ori + two_pi, ori) / two_pi;
      auto ori_index = ori_normalized * O;

      // Linear part of the patch normalization transform.
      auto T = Matrix2{};
      T(0, 0) = Halide::cos(theta);  T(0, 1) = Halide::sin(theta);
      T(1, 0) = -Halide::sin(theta); T(1, 1) = Halide::cos(theta);
      T /= Halide::Expr(bin_length_in_scale_unit * sigma);

      auto p = Vector2{};
      p(0) = r.x;
      p(1) = r.y;

      // Calculate the coordinates of the gradient in the reoriented normalized
      // patch.
      const auto Tp = T * p;
      auto Tp2 = Vector2{};
      Tp2(0) = Tp(0) - N / 2.f - 0.5f;
      Tp2(1) = Tp(1) - N / 2.f - 0.5f;

      // The weight of this gradient is:
      const auto weight = Halide::exp(-squared_norm(Tp2) /  //
                                      (2 * Halide::pow(N / 2.f, 2)));

      // Now the SIFT bin h(i, j, o) is calculated as:
      auto dx = Halide::abs(Halide::cast<float>(j) - Tp(0));
      auto dy = Halide::abs(Halide::cast<float>(i) - Tp(1));
      auto dori = Halide::abs(Halide::cast<float>(o) - ori_index);
      auto wx = Halide::select(dx < 1, 1 - dx, 0);
      auto wy = Halide::select(dy < 1, 1 - dy, 0);
      auto wo = Halide::select(dori < 1, 1 - dori, 0);

      return sum(wx * wy * wo * mag);
    }
  };

}  // namespace DO::Shakti::HalideBackend
