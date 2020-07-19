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
    static constexpr auto two_pi = static_cast<float>(2 * M_PI);

    // We construct a grid of NxN histogram of gradients.
    const std::int32_t N{4};
    // Each histogram has 8 bins.
    const std::int32_t O{8};

    const float bin_length_in_scale_unit{3.f};
    const float max_bin_value{0.2f};

    // Each image has a resolution scale (in pixels) associated to it.
    float scale_upper_bound;

    auto patch_radius_bound(const Halide::Expr& scale_upper_bound) const
    {
      return Halide::cast<std::int32_t>(
          Halide::round(bin_length_in_scale_unit * scale_upper_bound * (N + 1) /
                        2.f * std::sqrt(2.f)));
    }

    auto patch_radius(const Halide::Expr& scale) const
    {
      return Halide::cast<int>(Halide::round(bin_length_in_scale_unit * scale *
                                             (N + 1) / 2.f * std::sqrt(2.f)));
    }

    auto compute_bin_value(const Halide::Var& i,             //
                           const Halide::Var& j,             //
                           const Halide::Var& o,             //
                           const Halide::Func& grad_mag_fn,  //
                           const Halide::Func& grad_ori_fn,  //
                           const Halide::Expr& x,            //
                           const Halide::Expr& y,            //
                           const Halide::Expr& scale,        //
                           const Halide::Expr& scale_max,    //
                           const Halide::Expr& theta) const  //
    {
      // Calculate the radius upper-bound.
      const auto r_max = patch_radius_bound(scale_max);
      // Calculate the radius of the actual patch.
      const auto r_actual = patch_radius(scale);

      // Define the reduction domain.
      auto r = Halide::RDom(-r_max, 2 * r_max + 1, -r_max, 2 * r_max + 1);
      // The actual shape of the reduction domain is:
      r.where(Halide::abs(r.x) < r_actual &&  //
              Halide::abs(r.y) < r_actual);

      // The gradient magnitude is:
      const auto xi = Halide::round(x);
      const auto yi = Halide::round(y);
      const auto mag = grad_mag_fn(xi + r.x, yi + r.y);

      // The orientation in the reoriented patch is:
      const auto ori_shifted = grad_ori_fn(xi + r.x, yi + r.y) - theta;
      // We have:
      // -π < ori < π
      // -π < -theta < π
      // -2π < ori - theta < 2π
      // So we make sure that the orientation is [0, 2π[ as follows:
      const auto ori = Halide::select(ori_shifted < 0,       //
                                      ori_shifted + two_pi,  //
                                      ori_shifted);          //

      const auto ori_normalized = ori / two_pi;
      const auto ori_index = ori_normalized * O;

      // Linear part of the patch normalization transform.
      const Halide::Expr bin_length_in_pixels = bin_length_in_scale_unit * scale;
      auto T = Matrix2{};
      T(0, 0) = Halide::cos(theta);  T(0, 1) = Halide::sin(theta);
      T(1, 0) = -Halide::sin(theta); T(1, 1) = Halide::cos(theta);
      T /= bin_length_in_pixels;

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

      return Halide::sum(wx * wy * wo * mag);
    }

    auto normalize(const Halide::Func& h,       //
                   const Halide::Var& i,        //
                   const Halide::Var& j,        //
                   const Halide::Var& o,        //
                   const Halide::Var& k) const  //
    {
      auto r = Halide::RDom(0, N, 0, N, 0, O);
      auto contrast_norm = Halide::Func{"contrast_change_norm"};
      contrast_norm(k) = Halide::sqrt(Halide::sum(Halide::pow(h(r.x, r.y, r.z, k), 2)));
      contrast_norm.compute_root();

      auto h_contrast_invariant = Halide::Func{"h_contrast_invariant"};
      h_contrast_invariant(i, j, o, k) = h(i, j, o, k) / contrast_norm(k);
      h_contrast_invariant.compute_root();

      // Nonlinear illumination changes.
      auto h_clamped = Halide::Func{"h_clamped"};
      h_clamped(i, j, o, k) = Halide::min(h_contrast_invariant(i, j, o, k), 0.2f);

      auto illumination_norm = Halide::Func{"illumination_norm"};
      illumination_norm(k) = Halide::sqrt(Halide::sum(Halide::pow(h_clamped(r.x, r.y, r.z, k), 2)));
      illumination_norm.compute_root();

      auto h_illumination_invariant = Halide::Func{"h_illumination_invariant"};
      h_illumination_invariant(i, j, o, k) = h_clamped(i, j, o, k) /  //
                                             illumination_norm(k);
    }
  };

}  // namespace DO::Shakti::HalideBackend
